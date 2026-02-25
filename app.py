import os
import time
import warnings
import fitz  # PyMuPDF
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# Suppress torch/transformers FutureWarnings (cosmetic, no functional impact)
warnings.filterwarnings("ignore", category=FutureWarning)

# === ENV Setup ===
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INDEX_NAME = "pdf-qa-chatbot"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"

# === Streamlit UI Config ===
st.set_page_config(
    page_title="PDF ChatBot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Dark Theme CSS ===
st.markdown("""
<style>
/* ── Base ───────────────────────────────────────────────── */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* ── Sidebar ────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* ── Chat Messages ──────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

/* ── Chat Input ─────────────────────────────────────────── */
[data-testid="stChatInput"] textarea {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Buttons ────────────────────────────────────────────── */
.stButton > button {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover { background-color: #2ea043; }

/* ── File Uploader ──────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background-color: #21262d;
    border: 1px dashed #30363d;
    border-radius: 8px;
}

/* ── Divider ────────────────────────────────────────────── */
hr { border-color: #30363d; }

/* ── Info / Warning / Error boxes ──────────────────────── */
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

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
    from pinecone import ServerlessSpec  # deferred to avoid Streamlit lazy-import conflict
    pc = connect_pinecone()
    # Create index only if it doesn't exist yet
    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(index_name)
    # Always clear old vectors and upsert fresh ones for the current PDF
    index.delete(delete_all=True)
    pine_vectors = [{
        "id": f"chunk-{i}",
        "values": vector.tolist(),
        "metadata": {"text": text_chunks[i]}
    } for i, vector in enumerate(vectors)]
    # Upsert in batches of 100 to stay under Pinecone's 2MB request limit
    batch_size = 100
    for i in range(0, len(pine_vectors), batch_size):
        index.upsert(vectors=pine_vectors[i : i + batch_size])
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

    # Retry up to 3 times with exponential backoff on rate limit errors
    max_retries = 3
    for attempt in range(max_retries):
        try:
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
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)  # 2s, 4s, 8s
                time.sleep(wait)
            else:
                raise

# === Session State Init ===
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "index" not in st.session_state:
    st.session_state.index = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_chunks" not in st.session_state:
    st.session_state.pdf_chunks = 0

# === Sidebar ===
with st.sidebar:
    st.markdown("## 📄 PDF ChatBot")
    st.markdown("*Powered by **Llama 3.3 70B** via OpenRouter*")
    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("⚙️ Processing PDF..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())
                text = load_pdf_text("temp.pdf")
                chunks = split_text(text)
                embedder = load_embedder()
                vectors = get_embeddings(chunks, embedder)
                index = setup_pinecone(INDEX_NAME, vectors, chunks)

                st.session_state.index = index
                st.session_state.embedder = embedder
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_chunks = len(chunks)
                st.session_state.pdf_processed = True
                st.session_state.messages = []  # reset chat for new PDF

    if st.session_state.pdf_processed:
        st.success("✅ PDF Ready")
        st.markdown(f"**📄** `{st.session_state.pdf_name}`")
        st.markdown(f"**🔢 Chunks indexed:** {st.session_state.pdf_chunks}")
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("👆 Upload a PDF to begin")

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown("""
1. Upload any PDF
2. Text is extracted with PyMuPDF
3. Chunks are embedded via SentenceTransformers
4. Embeddings stored in Pinecone vector DB
5. Your question retrieves the top 5 relevant chunks
6. Llama 3.3 70B generates a grounded answer
        """)
    st.divider()
    st.markdown("Built by **Rohit Sonawane**")
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/rohit-sonawane245/) · [💻 GitHub](https://github.com/RohitDSonawane)")

# === Main Chat Area ===
st.markdown("## 🤖 Chat with your PDF")

if not st.session_state.pdf_processed:
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 0; color: #8b949e;'>
            <h3>👈 Upload a PDF from the sidebar to get started</h3>
            <p>Ask anything about your document — summaries, facts, explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Render full chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = ask_question(
                        st.session_state.index,
                        prompt,
                        st.session_state.embedder,
                    )
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except RateLimitError:
                    st.warning(
                        "⚠️ Model is temporarily rate-limited (free tier). "
                        "Please wait 10–15 seconds and try again."
                    )
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
