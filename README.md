
# AskYourPDF: FastAPI-Powered RAG Chatbot with Groq + FAISS

This project implements an end-to-end **Retrieval-Augmented Generation (RAG)** pipeline that lets users **upload PDFs and ask questions** about their content. It’s powered by:

* ⚙️ **FastAPI** for a blazing-fast backend
* 🧠 **Groq (LLaMA 3)** for intelligent language understanding
* 📄 **PDF Upload + Chunking** for ingesting documents
* 📚 **FAISS + HuggingFace Embeddings** for vector search
* 🔁 **History-Aware Rewriting** to make follow-ups context-rich

> Don’t just search your PDFs — *converse* with them intelligently.

---

**Check out the blog for detailed understanding :** [RAG chatbot using Langchain](https://medium.com/@avenkatesh0610)

---

## 🚀 Features

* 📄 Upload and embed your PDFs
* 🧠 Ask questions, get answers from your content
* 🔍 If no context is found, intelligently rewrite the query
* ✨ Powered by Groq for ultra-low latency responses
* 🔗 Chunked PDF text stored using FAISS for fast retrieval
* 🔧 Simple FastAPI backend and minimal chatbot UI (HTML)

---

## 📂 Project Structure

```
├── app.py                  # FastAPI backend + RAG logic
├── chatbot_ui.html         # Frontend chat interface
├── temp_uploads/           # Folder to store uploaded PDFs
├── vectors/                # Local FAISS index + embeddings
├── requirements.txt        # All Python dependencies
```

---

## 🔄 RAG Chatbot Flow

<img width="1500" height="511" alt="image" src="https://github.com/user-attachments/assets/78cc543d-6998-44cb-8147-b4680edf0c52" />

<img width="1816" height="521" alt="image" src="https://github.com/user-attachments/assets/e6084540-5ad4-4e9a-88ed-e09cdcc70f36" />

---

## ⚙️ Local Setup Instructions

Follow these steps to run the project locally:

```bash
**1. Clone the repository**

git clone https://github.com/Venkatesh0610/RAG-chatbot-using-own-pdf.git
cd RAG-chatbot-using-own-pdf

**2. Create and activate virtual environment**

python -m venv venv
source venv/bin/activate    # For Linux/macOS
venv\Scripts\activate       # For Windows

**3. Install required dependencies**

pip install -r requirements.txt

**4. Start the FastAPI server**

uvicorn app:app --reload --port 8080

**5. Open the chatbot UI**

Visit http://localhost:8080/ in your browser
```

---

## 💬 Example

> **User**: What are transformers?

> **System**: Transformers are neural network architectures based on self-attention mechanisms, widely used in NLP tasks such as translation, summarization, and question-answering.

---


## 🙌 Conclusion

In this project, we built an end-to-end PDF Question-Answering RAG system using FastAPI, FAISS, HuggingFace Embeddings, and Groq's blazing-fast LLaMA3 model. From PDF upload to chunking, embedding, retrieval, and final response generation — every step is streamlined and modular.

If you liked this, give it a **star**,and follow my [Blog](https://medium.com/@avenkatesh0610) and [Youtube](https://www.youtube.com/@avenkatesh0610) for more.
