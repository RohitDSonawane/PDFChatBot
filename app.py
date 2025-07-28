import os
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline
from pinecone import Pinecone, ServerlessSpec

# === ENV Setup ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "pdf-qa-chatbot"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Streamlit UI Config ===
st.set_page_config(page_title="📄 PDF ChatBot", layout="wide")
st.title("📄 Chat with your PDF!")

# === Caching resources ===
@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def load_qa_pipeline():
    return pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)

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

# === QA Answer Generation ===
def ask_question(index, query, embedder, qa_pipeline):
    query_vector = embedder.encode([query])[0].tolist()
    results = index.query(vector=query_vector, top_k=5, include_metadata=True)
    context = "\n".join([match["metadata"]["text"] for match in results["matches"]])
    prompt = f"Answer the question using the context below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    result = qa_pipeline(prompt)
    return result[0]["generated_text"]

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
        qa_pipeline = load_qa_pipeline()

    st.success("✅ PDF Processed! Ask your question below.")

    user_input = st.text_input("❓ Ask a question about your PDF:")
    if user_input:
        with st.spinner("💬 Generating answer..."):
            answer = ask_question(index, user_input, embedder, qa_pipeline)
        st.markdown(f"**Answer:** {answer}")
