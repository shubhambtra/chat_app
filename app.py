from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import faiss
import numpy as np
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client (NO hardcoded key)
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Please configure it in Railway Variables."
    )

client = OpenAI(api_key=api_key)


app = Flask(__name__)

# ===============================
# CHAT STORAGE
# ===============================
chat_messages = []

# ===============================
# VECTOR STORAGE
# ===============================
VECTOR_PATH = "data/vectors.index"
DOC_PATH = "data/docs.json"

os.makedirs("data", exist_ok=True)

index = None
documents = []

# Load existing vectors safely
if os.path.exists(VECTOR_PATH) and os.path.exists(DOC_PATH):
    try:
        index = faiss.read_index(VECTOR_PATH)
        with open(DOC_PATH, "r") as f:
            documents = json.load(f)
        print("✅ Vector index loaded")
    except Exception as e:
        print("⚠️ Failed to load index:", e)
        index = None
        documents = []

# ===============================
# ROUTES
# ===============================
@app.route("/")
def home():
    return "Chat App is running"

@app.route("/customer")
def customer():
    return render_template("customer.html")

@app.route("/sales")
def sales():
    return render_template("sales.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/send_customer", methods=["POST"])
def send_customer():
    msg = request.json.get("message")
    chat_messages.append({"sender": "Customer", "text": msg})
    return jsonify({"status": "ok"})

@app.route("/send_sales", methods=["POST"])
def send_sales():
    msg = request.json.get("message")
    chat_messages.append({"sender": "Sales", "text": msg})
    return jsonify({"status": "ok"})

@app.route("/messages")
def messages():
    return jsonify(chat_messages)

# ===============================
# DOCUMENT UPLOAD
# ===============================
@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    global index, documents

    text = request.form.get("content")
    if not text:
        return "No content provided", 400

    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    vec = np.array([embedding]).astype("float32")

    if index is None:
        index = faiss.IndexFlatL2(len(vec[0]))

    index.add(vec)
    documents.append(text)

    faiss.write_index(index, VECTOR_PATH)
    with open(DOC_PATH, "w") as f:
        json.dump(documents, f)

    return "✅ Document uploaded"

# ===============================
# VECTOR ANALYSIS
# ===============================
@app.route("/analyze", methods=["POST"])
def analyze():
    message = request.json.get("message", "").lower().strip()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    greetings = ["hi", "hello", "hey"]
    if message in greetings:
        return jsonify({
            "suggested_reply": "Hello! How can I help you?",
            "interest_level": "Low",
            "conversion_percentage": 10,
            "objection": "None",
            "next_action": "Ask needs"
        })

    if index is None or not documents:
        return jsonify({
            "suggested_reply": "No data available yet.",
            "interest_level": "Low",
            "conversion_percentage": 0,
            "objection": "No data",
            "next_action": "Upload documents"
        })

    q_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=message
    ).data[0].embedding

    q_vec = np.array([q_embedding]).astype("float32")
    D, I = index.search(q_vec, 1)

    if D[0][0] > 1.2:
        return jsonify({
            "suggested_reply": "Could you clarify your question?",
            "interest_level": "Medium",
            "conversion_percentage": 30,
            "objection": "Low relevance",
            "next_action": "Clarify"
        })

    answer = documents[I[0][0]]

    return jsonify({
        "suggested_reply": answer,
        "interest_level": "High",
        "conversion_percentage": int(100 / (1 + D[0][0])),
        "objection": "None",
        "next_action": "Proceed"
    })

# ===============================
# ENTRY POINT (Railway-safe)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
