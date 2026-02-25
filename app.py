import warnings
import streamlit as st
from dotenv import load_dotenv
from openai import RateLimitError

from config import INDEX_NAME
from pdf_utils import load_pdf_text, split_text
from embedder import load_embedder, get_embeddings
from vector_store import setup_index, query_index
from llm import get_answer

# Suppress torch/transformers FutureWarnings (cosmetic, no functional impact)
warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

# === Streamlit UI Config ===
st.set_page_config(
    page_title="PDF ChatBot",
    page_icon=":page_facing_up:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === Dark Theme CSS ===
st.markdown("""
<style>
/* -- Base ---------------------------------------------- */
.stApp { background-color: #0d1117; color: #e6edf3; }

/* -- Sidebar ------------------------------------------- */
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* -- Chat Messages ------------------------------------- */
[data-testid="stChatMessage"] {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
}

/* -- Chat Input ---------------------------------------- */
[data-testid="stChatInput"] textarea {
    background-color: #21262d !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* -- Buttons ------------------------------------------- */
.stButton > button {
    background-color: #238636;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stButton > button:hover { background-color: #2ea043; }

/* -- File Uploader ------------------------------------- */
[data-testid="stFileUploader"] {
    background-color: #21262d;
    border: 1px dashed #30363d;
    border-radius: 8px;
}

/* -- Divider ------------------------------------------- */
hr { border-color: #30363d; }

/* -- Info / Warning / Error boxes ---------------------- */
.stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


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
    st.markdown("## PDF ChatBot")
    st.markdown("*Powered by **Llama 3.3 70B** via OpenRouter*")
    st.divider()

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.pdf_name != uploaded_file.name:
            with st.spinner("Processing PDF..."):
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())
                text = load_pdf_text("temp.pdf")
                chunks = split_text(text)
                embedder = load_embedder()
                vectors = get_embeddings(chunks, embedder)
                index = setup_index(INDEX_NAME, vectors, chunks)

                st.session_state.index = index
                st.session_state.embedder = embedder
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.pdf_chunks = len(chunks)
                st.session_state.pdf_processed = True
                st.session_state.messages = []  # reset chat for new PDF

    if st.session_state.pdf_processed:
        st.success("PDF Ready")
        st.markdown(f"**File:** `{st.session_state.pdf_name}`")
        st.markdown(f"**Chunks indexed:** {st.session_state.pdf_chunks}")
        st.divider()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    else:
        st.info("Upload a PDF to begin")

    st.divider()
    with st.expander("How it works"):
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
    st.markdown("[LinkedIn](https://www.linkedin.com/in/rohit-sonawane245/) | [GitHub](https://github.com/RohitDSonawane)")

# === Main Chat Area ===
st.markdown("## Chat with your PDF")

if not st.session_state.pdf_processed:
    st.markdown(
        """
        <div style='text-align:center; padding: 80px 0; color: #8b949e;'>
            <h3>Upload a PDF from the sidebar to get started</h3>
            <p>Ask anything about your document - summaries, facts, explanations.</p>
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
                    query_vector = st.session_state.embedder.encode([prompt])[0].tolist()
                    context = query_index(st.session_state.index, query_vector)
                    answer = get_answer(context, prompt)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except RateLimitError:
                    st.warning(
                        "Model is temporarily rate-limited (free tier). "
                        "Please wait 10-15 seconds and try again."
                    )
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
