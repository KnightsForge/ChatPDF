import os
import fitz  # PyMuPDF
import chromadb  # Vector database
import hashlib
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Configure Google Gemini API
genai.configure(api_key="AIzaSyBVUwbfrssD8Qg3K2SWCdHhMtZw5YHJSmI")
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Sentence Transformer for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

@app.route("/")
def home():
    return render_template("index.html")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file page by page."""
    doc = fitz.open(pdf_path)
    texts = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        texts.append((page_num + 1, text))  # Store page number with text
    return texts


def store_pdf_in_chromadb(pdf_path, pdf_name):
    """Stores PDF content as embeddings in ChromaDB."""
    texts = extract_text_from_pdf(pdf_path)
    for page_num, text in texts:
        text_hash = hashlib.sha256(text.encode()).hexdigest()  # Unique ID
        embedding = embedding_model.encode(text).tolist()
        collection.add(
            ids=[text_hash],
            embeddings=[embedding],
            metadatas=[{"pdf_name": pdf_name, "page": page_num, "text": text}]
        )


def search_query_in_pdfs(query):
    """Searches the query in stored PDF embeddings and retrieves relevant chunks."""
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    return results["metadatas"][0] if "metadatas" in results else []


def generate_response(query, context):
    """Generates a response using Google Gemini API."""
    context_text = "\n".join([chunk["text"] for chunk in context])
    response = model.generate_content(f"Context:\n{context_text}\n\nUser Query: {query}")
    return response.text


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handles PDF uploads and processes them."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    store_pdf_in_chromadb(filepath, file.filename)
    return jsonify({"message": "File uploaded and processed successfully"})


@app.route("/query", methods=["POST"])
def query_pdf():
    """Handles user queries and returns responses."""
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    context = search_query_in_pdfs(query)
    if not context:
        return jsonify({"response": "No relevant information found."})

    response_text = generate_response(query, context)
    return jsonify({
        "response": response_text,
        "source": [{"pdf_name": c["pdf_name"], "page": c["page"]} for c in context]
    })


if __name__ == "__main__":
    app.run(debug=True)