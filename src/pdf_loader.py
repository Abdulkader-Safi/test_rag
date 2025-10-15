"""PDF loading and text extraction with OCR support"""

import warnings

import io
import hashlib
import pickle
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader

from .config import CACHE_DIR, MAX_WORKERS, OCR_IMAGE_MAX_SIZE

warnings.filterwarnings("ignore")


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
    if image.mode != "L":
        image = image.convert("L")

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
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                if cached_data["mtime"] == current_mtime:
                    print(f"[CACHE HIT] Loading from cache: {pdf_path.name}")
                    return cached_data["docs"]
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
        with open(cache_path, "wb") as f:
            pickle.dump({"mtime": current_mtime, "docs": docs}, f)
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
        future_to_pdf = {
            executor.submit(_load_single_pdf, pdf): pdf for pdf in pdf_files
        }
        for future in as_completed(future_to_pdf):
            try:
                docs = future.result()
                all_docs.extend(docs)
            except Exception:
                pass

    return all_docs
