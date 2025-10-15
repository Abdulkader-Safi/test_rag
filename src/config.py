"""Configuration settings for the PDF RAG system"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Directories ---
DATA_DIR = Path("./my_pdfs")  # put PDFs here
CACHE_DIR = Path("./.pdf_cache")  # Cache for extracted text

# --- Model Configuration ---
TOP_K = 3  # Number of documents to retrieve
OLLAMA_MODEL = "mistral"  # or "phi", "llama2", etc.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Performance Settings ---
MAX_WORKERS = 4  # Parallel processing workers
OCR_IMAGE_MAX_SIZE = 2000  # Max image dimension for OCR

# --- Text Splitting ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- LLM Settings ---
LLM_TEMPERATURE = 0
LLM_NUM_CTX = 2048  # Context window
LLM_NUM_PREDICT = 8112  # Max response length

# --- PostgreSQL Configuration ---
COLLECTION_NAME = "pdf_embeddings"
_connection_string = os.getenv("DATABASE_URL")

if not _connection_string:
    raise ValueError("DATABASE_URL environment variable not set")

CONNECTION_STRING: str = _connection_string
