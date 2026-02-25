"""
embedder.py - SentenceTransformer model loading and vector encoding.
"""

import warnings
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

# Suppress torch/transformers FutureWarnings (cosmetic, no functional impact)
warnings.filterwarnings("ignore", category=FutureWarning)


@st.cache_resource
def load_embedder() -> SentenceTransformer:
    """Load and cache the sentence embedding model (loaded once per session)."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def get_embeddings(chunks: list[str], embedder: SentenceTransformer) -> np.ndarray:
    """Encode a list of text chunks into dense embedding vectors."""
    return embedder.encode(chunks)
