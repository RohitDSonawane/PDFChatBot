"""
vector_store.py - Pinecone vector store: connection, upsert, and query.
"""

import os
import numpy as np
import streamlit as st
from pinecone import Pinecone

from config import (
    PINECONE_DIMENSION,
    PINECONE_METRIC,
    PINECONE_CLOUD,
    PINECONE_REGION,
    UPSERT_BATCH_SIZE,
    LLM_TOP_K,
)


@st.cache_resource
def connect_pinecone() -> Pinecone:
    """Create and cache the Pinecone client (one connection per session)."""
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def setup_index(index_name: str, vectors: np.ndarray, text_chunks: list[str]):
    """
    Create the Pinecone index if it doesn't exist, clear all existing vectors,
    then upsert the new PDF's chunks in batches to respect the 2MB request limit.

    Returns the Pinecone Index object ready for querying.
    """
    from pinecone import ServerlessSpec  # deferred - avoids Streamlit lazy-import conflict

    pc = connect_pinecone()

    if index_name not in [i.name for i in pc.list_indexes()]:
        pc.create_index(
            name=index_name,
            dimension=PINECONE_DIMENSION,
            metric=PINECONE_METRIC,
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    index = pc.Index(index_name)
    index.delete(delete_all=True)  # clear previous PDF's vectors

    pine_vectors = [
        {
            "id": f"chunk-{i}",
            "values": v.tolist(),
            "metadata": {"text": text_chunks[i]},
        }
        for i, v in enumerate(vectors)
    ]

    for i in range(0, len(pine_vectors), UPSERT_BATCH_SIZE):
        index.upsert(vectors=pine_vectors[i : i + UPSERT_BATCH_SIZE])

    return index


def query_index(index, query_vector: list[float]) -> str:
    """
    Retrieve the top-k most relevant text chunks for a query vector and
    return them joined as a single context string.
    """
    results = index.query(vector=query_vector, top_k=LLM_TOP_K, include_metadata=True)
    return "\n".join([match["metadata"]["text"] for match in results["matches"]])
