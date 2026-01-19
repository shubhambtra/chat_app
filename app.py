from flask import Flask, render_template, request, jsonify
from openai import OpenAI
import faiss
import numpy as np
import json, os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

# SAFE LOAD (ONLY IF VALID)
if os.path.exists(VECTOR_PATH) and os.path.exists(DOC_PATH):
    try:
        index = faiss.read_index(VECTOR_PATH)
        with open(DOC_PATH, "r") as f:
            documents = json.load(f)
        print("✅ Vector index loaded")
    except:
        print("⚠️ Corrupt index detected, rebuilding")
        index = None
        documents = []


# ===============================
# ROUTES
# ===============================
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
# DOCUMENT UPLOAD (SAFE)
# ===============================
@app.route("/upload_doc", methods=["POST"])
def upload_doc():
    global index, documents

    text = request.form.get("content")
    if not text:
        return "No content provided", 400

    # Create embedding (ONE TIME)
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

    vec = np.array([emb]).astype("float32")

    # CREATE INDEX IF FIRST DOC
    if index is None:
        index = faiss.IndexFlatL2(len(vec[0]))
        print("🆕 New FAISS index created")

    index.add(vec)
    documents.append(text)

    # SAVE SAFELY
    faiss.write_index(index, VECTOR_PATH)
    with open(DOC_PATH, "w") as f:
        json.dump(documents, f)

    return "✅ Document stored successfully"


# ===============================
# VECTOR SEARCH (NO CHATGPT)
# ===============================
@app.route("/analyze", methods=["POST"])
def analyze():
    message = request.json.get("message").lower().strip()

    # ===============================
    # 1️⃣ INTENT FILTER (NO VECTOR)
    # ===============================
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if message in greetings:
        return jsonify({
            "suggested_reply": "Hello! How can I help you today?",
            "interest_level": "Low",
            "conversion_percentage": 10,
            "objection": "None",
            "next_action": "Ask needs"
        })

    if len(message) < 5:
        return jsonify({
            "suggested_reply": "Could you please tell me more about what you’re looking for?",
            "interest_level": "Low",
            "conversion_percentage": 15,
            "objection": "Unclear intent",
            "next_action": "Probe further"
        })

    # ===============================
    # 2️⃣ VECTOR SEARCH (MEANINGFUL)
    # ===============================
    if index is None or len(documents) == 0:
        return jsonify({
            "suggested_reply": "Our team will get back to you shortly.",
            "interest_level": "Low",
            "conversion_percentage": 0,
            "objection": "No data",
            "next_action": "Upload documents"
        })

    # Embed question
    q_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=message
    ).data[0].embedding

    q_vec = np.array([q_emb]).astype("float32")

    D, I = index.search(q_vec, k=1)

    # ⚠️ SIMILARITY THRESHOLD
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



if __name__ == "__main__":
    app.run(debug=True)
