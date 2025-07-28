# 📄 PDF ChatBot — Chat with Any PDF using AI!

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This is a powerful **AI-powered PDF ChatBot** where users can upload any PDF file and ask questions about its content. The app extracts, chunks, embeds the text, stores it in **Pinecone**, and uses **FLAN-T5** to generate smart answers.

🚀 Fast, lightweight, and built with love for agentic AI experimentation.

---

## ✨ Features

- 📄 Upload & parse PDF using PyMuPDF
- 🔍 Chunk text and embed with SentenceTransformers
- 🌲 Store embeddings in Pinecone Vector DB
- 💬 Ask questions and get answers via FLAN-T5 (Hugging Face)
- ⚡ Instant UI built with Streamlit

---

## 🧠 Tech Stack

- Python 3.10+
- [Streamlit](https://streamlit.io/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [Pinecone Vector DB](https://www.pinecone.io/)
- [Transformers (FLAN-T5)](https://huggingface.co/docs/transformers/index)
- [LangChain Text Splitter](https://docs.langchain.com/)

---

## 📂 Project Structure

```bash
pdfchatbot/
├── app.py             # Main Streamlit application
├── .env               # Hidden env file for API keys (should NOT be uploaded!)
├── requirements.txt   # Python dependencies
└── README.md          # You're reading it!
```


## ⚙️ Setup Instructions

1. **Clone the Repo**
   ```bash
   git clone https://github.com/your-username/pdfchatbot.git
   cd pdfchatbot
   ```
   
2.**Create .env**
   ```bash
PINECONE_API_KEY=your-pinecone-api-key
   ```

3.**Install Requirements**
   ```bash
pip install -requirements.txt
   ```
4.**Set up environment variables**
Create a .env file in the root directory:
 ```bash
PINECONE_API_KEY=your_actual_api_key
   ```

## ▶️ Run the App
   ```bash
streamlit run app.py
   ```

## 🧠 How It Works
1.User uploads a PDF → text is extracted using PyMuPDF

2.Text is chunked using LangChain’s RecursiveCharacterTextSplitter

3.Embeddings generated via SentenceTransformers (MiniLM)

4.Chunks uploaded to Pinecone

5.User enters a question → relevant chunks are retrieved

6.Flan-T5 model generates the answer based on context


## 👨‍💻 Author
Rohit Sonawane   
🌐 [LinkedIn-https://www.linkedin.com/in/rohit-sonawane245/]

## 🌟 Show your support
⭐️ Star this repo if you like it!   
💬 Feedback, ideas, or improvements are always welcome.   
🚀 This is the start of a journey to build crazy agentic AIs.   
