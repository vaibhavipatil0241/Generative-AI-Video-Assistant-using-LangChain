from flask import Flask, render_template, request, jsonify
import os
import re
import time
import threading
import traceback
from collections import OrderedDict

from dotenv import load_dotenv
load_dotenv()

from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from langchain_groq import ChatGroq


# -----------------------------------
# Flask App
# -----------------------------------

app = Flask(__name__)

CORS(app)


# -----------------------------------
# Configuration
# -----------------------------------

MAX_TRANSCRIPT_CHARS = 400000     # long videos are no longer cut at 100k
CHUNK_WORDS = 180                 # ~1000-1200 characters per chunk
CHUNK_OVERLAP_WORDS = 30
TOP_K = 5                         # chunks used for specific questions
OVERVIEW_CHUNKS = 8               # chunks spread across the video for general questions
MIN_SCORE = 0.03                  # below this the lexical match is treated as noise
MAX_CONTEXT_CHARS = 9000          # keeps the prompt small for Groq limits

MAX_SESSIONS = 20                 # small LRU cache -> safe for 512 MB RAM
SESSION_TTL_SECONDS = 60 * 60

ENGLISH_CODES = ("en", "en-US", "en-GB", "en-IN", "en-CA", "en-AU")

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# -----------------------------------
# Load Groq LLM
# -----------------------------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing")


llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=GROQ_MODEL,
    temperature=0.1,
    max_tokens=2048
)



# -----------------------------------
# Prompt Templates
# -----------------------------------

BASE_RULES = """You are a helpful AI video assistant that answers questions about a YouTube video.

The context below is taken from the video's transcript. The transcript is
auto-generated, has no punctuation and may contain spelling mistakes, so read
it carefully and interpret it sensibly.

Rules:
- Use ONLY the transcript context. Do not invent information.
- If the context truly does not contain the answer, say so clearly.
- Use simple professional language.
- Keep answers concise but informative and do not repeat yourself.

Language rules:
- Answer in English by default.
- If the user asks for another language (for example Hindi), answer completely
  in that language. For Hindi, use Devanagari script.
- If the user asks you to translate what the speaker says, translate the
  relevant transcript content into the requested language.
- Do not refuse a translation just because the transcript is already in that
  language; rewrite it naturally in the requested language.
- Preserve the speaker's original meaning.
"""

OVERVIEW_PROMPT_TEMPLATE = BASE_RULES + """
Use this format:

What is:
Write two concise sentences.

Key Points:
- Point 1
- Point 2
- Point 3
- Point 4

Summary:
Write two concise sentences.

Transcript context (in video order):
{context}

Question:
{question}

Answer:
"""

SPECIFIC_PROMPT_TEMPLATE = BASE_RULES + """
Answer the question directly first. Add a short bullet list only if it helps.

Transcript context (in video order):
{context}

Question:
{question}

Answer:
"""


# -----------------------------------
# Per-client index store (small LRU)
# -----------------------------------
# The old code kept ONE global index. With several gunicorn workers or several
# users, /load_video and /predict could hit different processes/users, so the
# chatbot answered from the wrong (or an empty) index.

_store = OrderedDict()
_store_lock = threading.Lock()


def get_param(name, default=""):
    """Read a parameter from form data, JSON body or query string."""

    value = request.form.get(name)

    if value is None:
        data = request.get_json(silent=True)
        if isinstance(data, dict):
            value = data.get(name)

    if value is None:
        value = request.args.get(name)

    if value is None:
        return default

    return str(value).strip()


def get_client_key():
    """Use session_id if the extension sends one, else fall back to the client IP."""

    session_id = get_param("session_id")

    if session_id:
        return "sid:" + session_id[:64]

    forwarded = request.headers.get("X-Forwarded-For", "")
    ip = forwarded.split(",")[0].strip() or request.remote_addr or "default"

    return "ip:" + ip


def save_index(key, index):

    with _store_lock:

        _store[key] = index
        _store.move_to_end(key)

        while len(_store) > MAX_SESSIONS:
            _store.popitem(last=False)


def load_index(key):

    with _store_lock:

        index = _store.get(key)

        if index is None:
            return None

        if time.time() - index["created"] > SESSION_TTL_SECONDS:
            _store.pop(key, None)
            return None

        _store.move_to_end(key)

        return index


# -----------------------------------
# Extract YouTube Video ID
# -----------------------------------

def extract_video_id(video_url):

    try:

        url_match = re.search(
            r"(?:v=|youtu\.be/|/shorts/|/embed/|/live/|/v/)([a-zA-Z0-9_-]{11})",
            video_url
        )

        if url_match:
            return url_match.group(1)

        # Allow a bare 11-character video id
        if re.fullmatch(r"[a-zA-Z0-9_-]{11}", video_url.strip()):
            return video_url.strip()

        return None

    except Exception:
        return None


# -----------------------------------
# Fetch Transcript (prefers English)
# -----------------------------------

def fetch_transcript_text(video_id):
    """
    Returns (text, language_code).

    Old code requested languages=["hi", "en"], so any video that had a Hindi
    transcript was indexed in Hindi while users asked in English. TF-IDF
    cannot match across languages, so nothing was ever found.
    Now: English first -> otherwise any transcript, translated to English
    when YouTube allows it.
    """

    ytt_api = YouTubeTranscriptApi()

    transcript_list = ytt_api.list(video_id)

    transcript = None

    try:
        transcript = transcript_list.find_transcript(list(ENGLISH_CODES))
    except Exception:
        transcript = None

    if transcript is None:

        available = list(transcript_list)

        if not available:
            raise RuntimeError("No transcripts available for this video")

        manual = [t for t in available if not t.is_generated]
        transcript = (manual or available)[0]

        if transcript.is_translatable:
            try:
                transcript = transcript.translate("en")
            except Exception:
                pass

    fetched = transcript.fetch()

    text = " ".join(snippet.text for snippet in fetched)

    # Remove [Music], [Applause] etc. and normalise whitespace
    text = re.sub(r"\[[^\]]{1,30}\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text, transcript.language_code


# -----------------------------------
# Split Transcript into Chunks
# -----------------------------------

def split_transcript(text):
    """Word-based chunking: never cuts a word in half."""

    words = text.split()

    if not words:
        return []

    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP_WORDS)

    result = []

    for start in range(0, len(words), step):

        piece = words[start:start + CHUNK_WORDS]

        if not piece:
            break

        result.append(" ".join(piece))

        if start + CHUNK_WORDS >= len(words):
            break

    return result


# -----------------------------------
# Tokenizer (English + Devanagari safe)
# -----------------------------------

TOKEN_RE = re.compile(r"[\w\u0900-\u097F]+", re.UNICODE)


def _stem(token):

    if not token.isascii():
        return token

    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[:-len(suffix)]

    return token


def tokenize(text):

    tokens = []

    for token in TOKEN_RE.findall(text.lower()):

        if len(token) < 2 or token in ENGLISH_STOP_WORDS:
            continue

        tokens.append(_stem(token))

    return tokens


# -----------------------------------
# Create TF-IDF Index
# -----------------------------------

def create_vector_db_from_youtube(video_url):
    """Returns (ok, message, index_or_None)."""

    try:

        video_id = extract_video_id(video_url)

        if not video_id:
            return False, "Invalid YouTube URL.", None

        print("Video ID:", video_id)

        full_text, language = fetch_transcript_text(video_id)

        if not full_text.strip():
            return False, "Transcript is empty.", None

        full_text = full_text[:MAX_TRANSCRIPT_CHARS]

        print("Transcript fetched (", language, ")", len(full_text), "chars")

        new_chunks = split_transcript(full_text)

        if not new_chunks:
            return False, "No transcript chunks created.", None

        print("Chunks created:", len(new_chunks))

        new_vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=None,
            lowercase=False,          # tokenize() already lowercases
            token_pattern=None,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=20000
        )

        new_chunk_vectors = new_vectorizer.fit_transform(new_chunks)

        index = {
            "video_id": video_id,
            "language": language,
            "chunks": new_chunks,
            "vectorizer": new_vectorizer,
            "vectors": new_chunk_vectors,
            "created": time.time()
        }

        print("TF-IDF index created successfully")

        return True, "OK", index

    except Exception as error:

        print("\n========== TRANSCRIPT ERROR ==========")
        traceback.print_exc()
        print("======================================\n")

        name = type(error).__name__

        if "Blocked" in name:
            message = (
                "YouTube blocked the transcript request from the server. "
                "Please try again later."
            )
        elif "Disabled" in name or "NoTranscript" in name:
            message = "This video has no transcript available."
        elif "Unavailable" in name:
            message = "This video is unavailable."
        else:
            message = (
                "Failed to load YouTube video. "
                "Check whether its transcript is available."
            )

        return False, message, None


# -----------------------------------
# Retrieve Relevant Chunks
# -----------------------------------

OVERVIEW_RE = re.compile(
    r"\b(about|summar\w*|overview|main (idea|point|topic)s?|"
    r"key (point|takeaway|idea)s?|takeaways?|tl;?dr|gist|"
    r"explain (this|the) video|discuss\w*|cover\w*|talk\w* about|"
    r"what (is|was|does) (this|the) video)\b",
    re.IGNORECASE
)


def is_overview_question(question):

    return bool(OVERVIEW_RE.search(question))


def evenly_spaced_indices(total, count):

    if total <= count:
        return list(range(total))

    step = (total - 1) / (count - 1)

    return sorted({int(round(i * step)) for i in range(count)})


def retrieve_relevant_chunks(index, question):
    """Returns chunk texts in video order."""

    chunks = index["chunks"]

    if is_overview_question(question):
        selected = evenly_spaced_indices(len(chunks), OVERVIEW_CHUNKS)
    else:
        question_vector = index["vectorizer"].transform([question])

        scores = cosine_similarity(
            question_vector,
            index["vectors"]
        ).flatten()

        ranked = scores.argsort()[::-1][:TOP_K]

        selected = sorted(
            int(i) for i in ranked if scores[i] >= MIN_SCORE
        )

        # No lexical match (vague wording, other language, etc.):
        # fall back to chunks spread over the whole video instead of
        # refusing. The LLM decides whether the answer is in there.
        if not selected:
            selected = evenly_spaced_indices(len(chunks), OVERVIEW_CHUNKS)

    result = []
    used = 0

    for i in selected:

        piece = chunks[i]

        if used + len(piece) > MAX_CONTEXT_CHARS and result:
            break

        result.append(piece)
        used += len(piece)

    return result


# -----------------------------------
# Home Route
# -----------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------
# Load YouTube Video
# -----------------------------------

@app.route("/load_video", methods=["POST"])
def load_video():

    video_url = get_param("video_url") or get_param("url")

    if not video_url:

        return jsonify({
            "status": "Please provide a YouTube video URL."
        }), 400

    ok, message, index = create_vector_db_from_youtube(video_url)

    if ok:

        save_index(get_client_key(), index)

        return jsonify({
            "status": "YouTube video loaded successfully!",
            "video_id": index["video_id"],
            "language": index["language"],
            "chunks": len(index["chunks"])
        })

    return jsonify({
        "status": message
    }), 400


# -----------------------------------
# Chat Route
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    user_message = get_param("message")

    if not user_message:

        return jsonify({
            "response": "Please enter a question."
        }), 400

    index = load_index(get_client_key())

    if index is None:

        return jsonify({
            "response": "Please load a YouTube video first."
        }), 400

    try:

        print("Question:", user_message)

        relevant_chunks = retrieve_relevant_chunks(index, user_message)

        context = "\n\n".join(relevant_chunks)

        template = (
            OVERVIEW_PROMPT_TEMPLATE
            if is_overview_question(user_message)
            else SPECIFIC_PROMPT_TEMPLATE
        )

        final_prompt = template.format(
            context=context,
            question=user_message
        )

        result = llm.invoke(final_prompt)

        answer = (result.content or "").strip()

        if not answer:
            answer = (
                "I couldn't generate an answer this time. "
                "Please try rephrasing your question."
            )

        print("Answer generated successfully")

        return jsonify({
            "response": answer
        })

    except Exception:

        print("\n========== PREDICTION ERROR ==========")
        traceback.print_exc()
        print("======================================\n")

        return jsonify({
            "response": "Sorry, something went wrong while generating the answer. Please try again."
        }), 500


# -----------------------------------
# Health Check
# -----------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "retrieval": "TF-IDF",
        "loaded_sessions": len(_store)
    })


# -----------------------------------
# Run App
# -----------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )