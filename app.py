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

    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",

    temperature=0.1
)

# -----------------------------------
# Prompt Template
# -----------------------------------

prompt = PromptTemplate(

    template="""
You are a helpful AI assistant.

Answer ONLY from the provided context.

You MUST answer in the EXACT format below.

what is :
<write 2 concise sentences>

Key Points:
• point 1
• point 2
• point 3
• point 4

Summary:
<write 2 concise sentences>
defination give only when there is it some definition in the context otherwise leave it blank

Rules:
- Translate the final answer into English before responding
- Never answer in Hindi
- Even if context is Hindi, output must be English only
- Use simple professional language
- Keep answers concise but informative
- Use bullet points only when useful
- Avoid unnecessary headings
- Do not repeat information
- Format answers cleanly

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