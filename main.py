import io
import os
import sys
import warnings

# Redirect stderr immediately to suppress all warnings
sys.stderr = open(os.devnull, 'w')

from pathlib import Path
from typing import List
import hashlib
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_SUPPRESS_WARNINGS"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress specific deprecation warnings
import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import PGVector
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings


# Load environment variables
load_dotenv()

# --- Config ---
DATA_DIR = Path("./my_pdfs")  # put PDFs here
CACHE_DIR = Path("./.pdf_cache")  # Cache for extracted text
TOP_K = 3  # Reduced from 4 for faster retrieval
OLLAMA_MODEL = "mistral"  # or "mistral", "phi", etc.
MAX_WORKERS = 4  # Parallel processing workers
OCR_IMAGE_MAX_SIZE = 2000  # Max image dimension for OCR

# PostgreSQL connection
COLLECTION_NAME = "pdf_embeddings"
_connection_string = os.getenv("DATABASE_URL")

if not _connection_string:
    raise ValueError("DATABASE_URL environment variable not set")

CONNECTION_STRING: str = _connection_string

# Singleton for embeddings model
_EMBEDDINGS_MODEL = None


def _get_image_hash(image_bytes: bytes) -> str:
    """Calculate hash for image deduplication"""
    return hashlib.md5(image_bytes).hexdigest()


def _preprocess_image(image: Image.Image) -> Image.Image:
    """Optimize image for faster OCR"""
    # Resize large images
    if max(image.size) > OCR_IMAGE_MAX_SIZE:
        ratio = OCR_IMAGE_MAX_SIZE / max(image.size)
        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to grayscale for faster OCR
    if image.mode != 'L':
        image = image.convert('L')

    return image


def _process_single_image(args):
    """Process a single image for OCR (for parallel processing)"""
    image_bytes, pdf_path, page_num = args
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = _preprocess_image(image)
        ocr_text = pytesseract.image_to_string(image)

        if ocr_text.strip():
            return Document(
                page_content=ocr_text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_num,
                    "type": "image_ocr",
                },
            )
    except Exception:
        pass
    return None


def extract_text_from_images(pdf_path: Path) -> List[Document]:
    """Extract text from images in PDF using OCR with parallel processing"""
    docs = []
    pdf_document = fitz.open(str(pdf_path))

    # Collect all images with deduplication
    image_tasks = []
    seen_hashes = set()

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images()

        for img in images:
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Skip duplicate images
                img_hash = _get_image_hash(image_bytes)
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                image_tasks.append((image_bytes, pdf_path, page_num))
            except Exception:
                pass

    pdf_document.close()

    # Process images in parallel
    if image_tasks:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results = executor.map(_process_single_image, image_tasks)
            docs = [doc for doc in results if doc is not None]

    return docs


def _get_pdf_cache_path(pdf_path: Path) -> Path:
    """Get cache file path for a PDF"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_hash = hashlib.md5(str(pdf_path).encode()).hexdigest()
    return CACHE_DIR / f"{pdf_hash}.pkl"


def _get_pdf_mtime(pdf_path: Path) -> float:
    """Get modification time of PDF"""
    return pdf_path.stat().st_mtime


def _load_single_pdf(pdf_path: Path) -> List[Document]:
    """Load a single PDF with caching"""
    cache_path = _get_pdf_cache_path(pdf_path)
    current_mtime = _get_pdf_mtime(pdf_path)

    # Check cache
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                if cached_data['mtime'] == current_mtime:
                    print(f"[CACHE HIT] Loading from cache: {pdf_path.name}")
                    return cached_data['docs']
        except Exception:
            pass

    # Load fresh
    print(f"[PROCESSING] Extracting text from: {pdf_path.name}")
    docs = []

    # Load text content
    loader = PyPDFLoader(str(pdf_path))
    text_docs = loader.load()
    docs.extend(text_docs)

    # Extract text from images using OCR
    image_docs = extract_text_from_images(pdf_path)
    docs.extend(image_docs)

    # Cache results
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({'mtime': current_mtime, 'docs': docs}, f)
        print(f"[CACHED] Saved to cache: {pdf_path.name}")
    except Exception:
        pass

    return docs


def load_pdfs(pdf_dir: Path) -> List[Document]:
    """Load PDFs with parallel processing and caching"""
    pdf_files = list(pdf_dir.glob("**/*.pdf"))

    if not pdf_files:
        return []

    # Process PDFs in parallel
    all_docs = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdf = {executor.submit(_load_single_pdf, pdf): pdf for pdf in pdf_files}
        for future in as_completed(future_to_pdf):
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception:
                pass

    return all_docs

def chunk_docs(docs: List[Document]) -> List[Document]:
    """Chunk documents with parallel processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
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

def get_embeddings():
    """Get embeddings model with singleton pattern"""
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDINGS_MODEL


def get_llm():
    return Ollama(
        model=OLLAMA_MODEL,
        temperature=0,
        num_ctx=2048,  # Reduced context window for faster inference
        num_predict=8112,  # Allow longer responses for structured output
    )

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

def make_qa_chain(vs: PGVector):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )


if __name__ == "__main__":
    import argparse
    from rich.console import Console
    from rich.panel import Panel

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--add-pdfs", action="store_true", help="Add PDFs from my_pdfs directory to vector store"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear all vectors from the database"
    )
    parser.add_argument(
        "-q", "--query", type=str, help="Ask a single question and exit"
    )
    args = parser.parse_args()

    console = Console()

    vs = None
    qa = None
    try:
        if args.clear:
            console.print("[bold yellow]Clearing vector store...[/bold yellow]")
            vs = ensure_index()
            vs.delete_collection()
            console.print("[bold green]Vector store cleared successfully.[/bold green]")
            sys.exit(0)

        if args.add_pdfs:
            add_pdfs_to_index()
            sys.exit(0)

        with console.status("[bold green]Initializing QA chain...[/bold green]"):
            vs = ensure_index()
            qa = make_qa_chain(vs)

        # Handle single query mode
        if args.query:
            for chunk in qa.stream({"query": args.query}):
                if "result" in chunk:
                    console.print(chunk["result"], end="", style="")

            console.print()
            sys.exit(0)

        console.print("[bold green]Starting interactive chat. Type 'exit' or press Ctrl+C to quit.[/bold green]")

        while True:
            try:
                query = console.input("[bold cyan]> [/bold cyan]")
                if query.lower() == 'exit':
                    break

                console.print("[bold green]Answer:[/bold green] ", end="")
                source_documents = []
                
                for chunk in qa.stream({"query": query}):
                    if "result" in chunk:
                        console.print(chunk["result"], end="", style="")
                    if "source_documents" in chunk:
                        source_documents.extend(chunk["source_documents"])
                
                console.print()

                if source_documents:
                    console.print("\n[bold]Source Documents:[/bold]")
                    for doc in source_documents:
                        console.print(Panel(
                            f"[cyan]{doc.metadata['source']}[/cyan] (Page {doc.metadata.get('page', 'N/A')})\n\n{doc.page_content}",
                            title=f"[bold yellow]Source[/bold yellow]",
                            border_style="yellow"
                        ))

            except KeyboardInterrupt:
                console.print("\n[bold red]Exiting...[/bold red]")
                break

    finally:
        # Explicitly clean up in reverse order
        if qa is not None:
            try:
                del qa
            except Exception:
                pass
        if vs is not None:
            try:
                del vs
            except Exception:
                pass