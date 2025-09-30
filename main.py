import io
import os
import warnings
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

# Suppress all warnings
warnings.filterwarnings("ignore")

from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Config ---
DATA_DIR = Path("./my_pdfs")  # put PDFs here
INDEX_DIR = Path("./vector_index")  # FAISS index
TOP_K = 4
OLLAMA_MODEL = "mistral"  # or "mistral", "phi", etc.


def extract_text_from_images(pdf_path: Path) -> List[Document]:
    """Extract text from images in PDF using OCR"""
    docs = []
    pdf_document = fitz.open(str(pdf_path))

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        images = page.get_images()

        for img_index, img in enumerate(images):
            try:
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]

                # Convert to PIL Image
                image = Image.open(io.BytesIO(image_bytes))

                # Perform OCR
                ocr_text = pytesseract.image_to_string(image)

                if ocr_text.strip():  # Only add if text was found
                    doc = Document(
                        page_content=ocr_text,
                        metadata={
                            "source": str(pdf_path),
                            "page": page_num,
                            "type": "image_ocr",
                        },
                    )
                    docs.append(doc)
            except Exception:
                pass

    pdf_document.close()
    return docs


def load_pdfs(pdf_dir: Path) -> List[Document]:
    docs = []
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    for pdf_path in pdf_files:
        # Load text content
        loader = PyPDFLoader(str(pdf_path))
        text_docs = loader.load()
        docs.extend(text_docs)

        # Extract text from images using OCR
        image_docs = extract_text_from_images(pdf_path)
        docs.extend(image_docs)

    return docs


def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0)


def ensure_index():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Check if index already exists
    if (INDEX_DIR / "index.faiss").exists():
        return FAISS.load_local(
            folder_path=str(INDEX_DIR),
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )

    # Index doesn't exist, need to build it
    docs = load_pdfs(DATA_DIR)
    if not docs:
        raise RuntimeError("No PDFs found in directory.")
    chunks = chunk_docs(docs)

    vs = FAISS.from_documents(chunks, get_embeddings())
    vs.save_local(str(INDEX_DIR))
    return vs


def make_qa_chain(vs: FAISS):
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )


def ask(qa, query: str):
    result = qa({"query": query})
    print(result["result"])


if __name__ == "__main__":
    import argparse
    import sys

    # Suppress stderr to hide all warnings
    sys.stderr = open(os.devnull, 'w')

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", required=True, help="Your question")
    parser.add_argument(
        "--rebuild", action="store_true", help="Rebuild the FAISS index"
    )
    args = parser.parse_args()

    if args.rebuild and INDEX_DIR.exists():
        for p in INDEX_DIR.glob("*"):
            p.unlink()

    vs = ensure_index()
    qa = make_qa_chain(vs)
    ask(qa, args.query)
