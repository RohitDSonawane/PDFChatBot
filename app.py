import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

# === ENV Setup ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INDEX_NAME = "pdf-qa-chatbot"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# === Streamlit UI Config ===
st.set_page_config(page_title="📄 PDF ChatBot", layout="wide")
st.title("📄 Chat with your PDF!")

# === Caching resources ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def connect_pinecone():
    return Pinecone(api_key=PINECONE_API_KEY)

# === PDF Text Extraction ===
def load_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# === Text Chunking ===
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# === Embed Chunks ===
def get_embeddings(chunks, embedder):
    vectors = embedder.encode(chunks)
    return vectors

# === Pinecone Upload ===
def setup_pinecone(index_name, vectors, text_chunks):
    pc = connect_pinecone()
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        index = pc.Index(index_name)
        pine_vectors = [{
            "id": f"chunk-{i}",
            "values": vector.tolist(),
            "metadata": {"text": text_chunks[i]}
        } for i, vector in enumerate(vectors)]
        index.upsert(vectors=pine_vectors)
    else:
        index = pc.Index(index_name)
    return index

# === QA Answer Generation via OpenRouter ===
def ask_question(index, query, embedder):
    query_vector = embedder.encode([query])[0].tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    response = client.chat.completions.create(
        model=OPENROUTER_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions strictly based on the provided PDF context. If the answer is not in the context, say so clearly.",
            },
            {
                "role": "user",
                "content": f"Context from PDF:\n{context}\n\nQuestion: {query}",
            },
        ],
    )
    return response.choices[0].message.content

# === Streamlit UI ===
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("📄 Processing your PDF..."):
        text = load_pdf_text("temp.pdf")
        chunks = split_text(text)
        embedder = load_embedder()
        vectors = get_embeddings(chunks, embedder)
        index = setup_pinecone(INDEX_NAME, vectors, chunks)

    st.success("✅ PDF Processed! Ask your question below.")

    user_input = st.text_input("❓ Ask a question about your PDF:")
    if user_input:
        with st.spinner("💬 Generating answer..."):
            answer = ask_question(index, user_input, embedder)
        st.markdown(f"**Answer:** {answer}")
