"""Vector store operations using PostgreSQL"""
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import (
    COLLECTION_NAME,
    CONNECTION_STRING,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MAX_WORKERS,
    EMBEDDING_MODEL,
    DATA_DIR
)
from .pdf_loader import _load_single_pdf


# Singleton for embeddings model
_EMBEDDINGS_MODEL = None


def get_embeddings():
    """Get embeddings model with singleton pattern"""
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
    return _EMBEDDINGS_MODEL


def chunk_docs(docs: List[Document]) -> List[Document]:
    """Chunk documents with parallel processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )

    # Split documents in parallel batches
    if len(docs) <= 10:
        return splitter.split_documents(docs)

    batch_size = max(1, len(docs) // MAX_WORKERS)
    batches = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    all_chunks = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(splitter.split_documents, batch) for batch in batches]
        for future in as_completed(futures):
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
            except Exception:
                pass

    return all_chunks


def ensure_index():
    """Initialize or connect to PostgreSQL vector store"""
    # Initialize connection to existing vectors
    vs = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=get_embeddings(),
    )

    return vs


def add_pdfs_to_index(pdf_paths: List[Path] | None = None):
    """Add new PDFs to the PostgreSQL vector store"""
    if pdf_paths is None:
        # Process all PDFs in directory
        pdf_paths = list(DATA_DIR.glob("**/*.pdf"))

    if not pdf_paths:
        print("No PDFs to process.")
        return

    # Load and process PDFs
    all_docs = []
    for pdf_path in pdf_paths:
        print(f"Processing: {pdf_path.name}")
        docs = _load_single_pdf(pdf_path)
        all_docs.extend(docs)

    if not all_docs:
        raise RuntimeError("No content extracted from PDFs.")

    # Chunk documents
    chunks = chunk_docs(all_docs)
    print(f"Adding {len(chunks)} chunks to vector store...")

    # Add to PostgreSQL
    vs = None
    try:
        vs = PGVector.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
        )
        print(f"Successfully added {len(chunks)} chunks from {len(pdf_paths)} PDFs.")
    finally:
        if vs is not None:
            try:
                del vs
            except Exception:
                pass
