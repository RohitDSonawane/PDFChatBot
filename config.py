"""
config.py - All tuneable constants in one place.
Change values here to affect the entire application.
"""

# Pinecone
INDEX_NAME: str = "pdf-qa-chatbot"
PINECONE_DIMENSION: int = 384
PINECONE_METRIC: str = "cosine"
PINECONE_CLOUD: str = "aws"
PINECONE_REGION: str = "us-east-1"
UPSERT_BATCH_SIZE: int = 100   # max vectors per Pinecone request (~2MB limit)

# Embeddings
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"

# LLM
OPENROUTER_MODEL: str = "meta-llama/llama-3.3-70b-instruct:free"
LLM_TOP_K: int = 5             # number of chunks retrieved per query
LLM_MAX_RETRIES: int = 3       # retry attempts on 429 rate-limit errors

# Text splitting
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50
