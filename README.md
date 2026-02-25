# PDF ChatBot - Chat with Any PDF using AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange)](https://streamlit.io)
[![Llama](https://img.shields.io/badge/LLM-Llama%203.3%2070B-blueviolet)](https://openrouter.ai)
[![Pinecone](https://img.shields.io/badge/Vector%20DB-Pinecone-green)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **AI-powered PDF ChatBot** with a dark chat UI. Upload any PDF, ask questions in a conversational interface, and get grounded answers powered by **Llama 3.3 70B** via OpenRouter.

> Built as a portfolio project - fast, lightweight, zero local GPU required.

---

## Features

- Upload & parse any PDF with PyMuPDF
- Smart text chunking via LangChain
- Semantic embeddings using `all-MiniLM-L6-v2`
- Vector storage & retrieval with Pinecone (serverless)
- Answers from **Llama 3.3 70B** via OpenRouter (free tier)
- Persistent chat history with full conversation UI
- Dark GitHub-inspired theme
- Auto-retry on rate limits, batch upsert for large PDFs

---

## Architecture

```
PDF Upload
    |
    v
PyMuPDF -- extract raw text
    |
    v
LangChain RecursiveCharacterTextSplitter -- chunk (500 tokens, 50 overlap)
    |
    v
SentenceTransformers all-MiniLM-L6-v2 -- embed chunks -> 384-dim vectors
    |
    v
Pinecone Serverless -- store & query vectors (top-5 retrieval)
    |
    v
OpenRouter -> Llama 3.3 70B Instruct -- generate grounded answer
    |
    v
Streamlit Dark Chat UI -- display in conversation thread
```

---

## Tech Stack

| Layer              | Technology                                                                                          |
| ------------------ | --------------------------------------------------------------------------------------------------- |
| **Frontend**       | [Streamlit](https://streamlit.io/) with custom dark CSS                                             |
| **PDF Parsing**    | [PyMuPDF](https://pymupdf.readthedocs.io/)                                                          |
| **Text Splitting** | [LangChain](https://docs.langchain.com/) RecursiveCharacterTextSplitter                             |
| **Embeddings**     | [SentenceTransformers](https://www.sbert.net/) all-MiniLM-L6-v2                                     |
| **Vector DB**      | [Pinecone](https://pinecone.io/) Serverless (free tier)                                             |
| **LLM**            | [Llama 3.3 70B](https://openrouter.ai/meta-llama/llama-3.3-70b-instruct:free) via OpenRouter (free) |
| **API Client**     | [OpenAI Python SDK](https://github.com/openai/openai-python)                                        |

---

## Project Structure

```
app/
├── app.py                  # Streamlit UI entry point
├── config.py               # All constants (model names, chunk sizes, etc.)
├── pdf_utils.py            # PDF extraction and text chunking
├── embedder.py             # SentenceTransformer loading and vector encoding
├── vector_store.py         # Pinecone connect, upsert, and query
├── llm.py                  # OpenRouter API call with retry logic
├── .streamlit/
│   └── config.toml         # Dark theme + server config for Streamlit Cloud
├── .env.example            # Template -- copy to .env and fill in keys
├── .gitignore              # Excludes .env, .venv, temp.pdf, __pycache__
├── requirements.txt        # Minimal direct dependencies (7 packages)
└── README.md               # This file
```

---

## Local Setup

### 1. Clone the repo

```bash
git clone https://github.com/RohitDSonawane/PDFChatBot.git
cd PDFChatBot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Then edit `.env` and add your keys:

```env
PINECONE_API_KEY=your-pinecone-api-key
OPENROUTER_API_KEY=your-openrouter-api-key
```

- Get Pinecone key: [app.pinecone.io](https://app.pinecone.io) -> API Keys
- Get OpenRouter key (free): [openrouter.ai](https://openrouter.ai) -> Keys

### 5. Run the app

```bash
streamlit run app.py
```

---

## Deploy on Streamlit Community Cloud (Free)

1. Push this repo to GitHub (public)
2. Go to [share.streamlit.io](https://share.streamlit.io) -> sign in with GitHub
3. Click **New app** -> select this repo -> set `app.py` as entry point
4. Go to **Advanced settings -> Secrets** and add:

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
OPENROUTER_API_KEY = "your-openrouter-api-key"
```

5. Click **Deploy** -- live in ~2 minutes

---

## Author

**Rohit Sonawane**

- [LinkedIn](https://www.linkedin.com/in/rohit-sonawane245/)
- [GitHub](https://github.com/RohitDSonawane)

---

Star this repo if you found it useful! Feedback, issues, and PRs are welcome.
