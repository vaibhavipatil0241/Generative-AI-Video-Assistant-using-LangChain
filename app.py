from flask import Flask, render_template, request, jsonify
import os

from dotenv import load_dotenv
load_dotenv()

from flask_cors import CORS
from youtube_transcript_api import YouTubeTranscriptApi

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_groq import ChatGroq


# -----------------------------------
# Flask App
# -----------------------------------

app = Flask(__name__)

CORS(app)
# -----------------------------------
# Load Embeddings
# -----------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# -----------------------------------
# Load Existing FAISS DB (Optional)
# -----------------------------------

try:

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

except:

    db = None

# -----------------------------------
# Load Groq LLM
# -----------------------------------
import os
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

# -----------------------------------
# Prompt Template
# -----------------------------------



prompt = PromptTemplate(

    template="""
You are a helpful AI video assistant.

Answer ONLY from the provided video transcript context.
Do not make up information.

You MUST follow the exact output format below:

Definition:
<Write a definition in 1-2 sentences only if a definition is available in the context. Otherwise, leave it blank.>

What is:
<Write 2 concise sentences explaining the answer.>

Key Points:
• Point 1
• Point 2
• Point 3
• Point 4

Summary:
<Write 2 concise sentences summarizing the answer.>

LANGUAGE RULES:
- Identify the language requested by the user.
- If the user asks for Hindi, respond completely in Hindi using Devanagari script.
- If the user asks for English, respond completely in English.
- If the user requests another language, respond in that language.
- If the user asks to translate what the speaker says, translate the relevant transcript content into the requested language.
- Never force English when the user explicitly requests another language.
- Preserve the original meaning of the speaker.

ANSWER RULES:
- Use simple and professional language.
- Keep answers concise but informative.
- Use bullet points for key points.
- Avoid unnecessary repetition.
- Do not add information outside the provided context.
- If the context does not contain the answer, clearly mention that the information is not available in the transcript.

TRANSLATION RULES:
- If the user asks to translate the video into Hindi, translate the relevant transcript content into Hindi.
- Even if the transcript is already in Hindi, translate any English words or sentences into Hindi wherever possible.
- If the transcript contains both Hindi and English (Hinglish), convert it into natural, fluent Hindi using Devanagari script.
- Do not say that a translation is unavailable just because the transcript is already in Hindi.
- Do not merely describe the transcript language. Provide the translated content.
- Preserve the speaker's original meaning.

Context:
{context}

Question:
{question}

Answer:
""",

    input_variables=["context", "question"]
)

# -----------------------------------
# Create Vector DB from YouTube
# -----------------------------------

def create_vector_db_from_youtube(video_url):

    global db

    try:

        # Handle different URL formats

        if "watch?v=" in video_url:

            video_id = video_url.split("watch?v=")[1].split("&")[0]

        elif "youtu.be/" in video_url:

            video_id = video_url.split("youtu.be/")[1].split("?")[0]

        else:

            return False

        print("Video ID:", video_id)

        ytt_api = YouTubeTranscriptApi()

        fetched_transcript = ytt_api.fetch(
         video_id,
          languages=['hi', 'en']
)

        full_text = " ".join(
         [snippet.text for snippet in fetched_transcript]
          ) 
     

        print("Transcript fetched successfully")

        # Split text

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=50
        )

        chunks = splitter.split_text(full_text)

        print("Chunks created:", len(chunks))

        # Create FAISS DB

        db = FAISS.from_texts(
            chunks,
            embeddings
        )

        # Save DB

        db.save_local("faiss_index")

        print("FAISS database created")

        return True

    except Exception as e:

     import traceback

     print("\n========== FULL ERROR ==========")

     traceback.print_exc()

     print("================================\n")

     return False

# -----------------------------------
# Home Route
# -----------------------------------

@app.route('/')
def home():

    return render_template('index.html')

# -----------------------------------
# Load YouTube Video
# -----------------------------------

@app.route('/load_video', methods=['POST'])
def load_video():

    video_url = request.form['video_url']

    success = create_vector_db_from_youtube(
        video_url
    )

    if success:

        return jsonify({
            "status": "YouTube video loaded successfully!"
        })

    else:

        return jsonify({
            "status": "Failed to load YouTube video."
        })

# -----------------------------------
# Chat Route
# -----------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    global db

    if db is None:

        return jsonify({
            "response": "Please load a YouTube video first."
        })

    user_message = request.form['message']

    print("Question:", user_message)

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(

        llm=llm,

        retriever=db.as_retriever(
            search_kwargs={"k": 3}
        ),

        chain_type_kwargs={
            "prompt": prompt
        }
    )

    # Generate answer
    result = qa_chain.invoke({
        "query": user_message
    })

    print("Answer generated")

    return jsonify({
        "response": result["result"]
    })

# -----------------------------------
# Run App
# -----------------------------------

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )