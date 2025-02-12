import os
import fitz  # PyMuPDF
import chromadb  # Vector database
import hashlib
from flask import Flask, request, jsonify, render_template
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import os
import markdown2

API_KEY = os.environ['GEMINI_API_KEY']
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize ChromaDB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks")

@app.route("/")
def home():
    return render_template("index.html")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    texts = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text("text")
        texts.append((page_num + 1, text))  # Store page number with text
    return texts

@app.route("/clear_db", methods=["POST"])
def clear_db():
    """Clears all documents from the ChromaDB collection."""
    chroma_client.delete_collection(name="pdf_chunks")
    global collection
    collection = chroma_client.get_or_create_collection(name="pdf_chunks")
    return jsonify({"message": "Database cleared successfully"})

def store_pdf_in_chromadb(pdf_path, pdf_name):
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
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=7)
    return results["metadatas"][0] if "metadatas" in results else []


def generate_response(query, context):
    context_text = "\n".join([chunk["text"] for chunk in context])
    response = model.generate_content(f"Context:\n{context_text}\n\nUser Query: {query}")
    response_html = markdown2.markdown(response.text)
    return response_html


@app.route("/upload", methods=["POST"])
def upload_pdf():
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
    data = request.json
    query = data.get("query", "")

    if not query:
        return jsonify({"error": "Query cannot be empty"}), 400

    context = search_query_in_pdfs(query)
    if not context:
        return jsonify({"response": "No relevant information found."})

    response_html = generate_response(query, context)
    return jsonify({
        "response": response_html,
        "source": [{"pdf_name": c["pdf_name"], "page": c["page"]} for c in context]
    })


if __name__ == "__main__":
    app.run(debug=True)
