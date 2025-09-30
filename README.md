# PDF RAG System

A Retrieval-Augmented Generation (RAG) system that processes PDF documents with OCR support and enables question-answering using local LLMs via Ollama.

![PDF RAG SYSTEM](https://abdulkadersafi.com/storage/158/01K6E35JVPR2MCC56CRHNWRNDP.png)

> Read Blog Article [Here](https://abdulkadersafi.com/blog/retrieval-augmented-generation-rag-with-local-pdfs-and-ollama-a-developers-guide)

## Features

- **PDF Text Extraction**: Extracts text from PDF documents using PyMuPDF
- **OCR Support**: Extracts text from images within PDFs using Tesseract OCR
- **Parallel Processing**: Processes multiple PDFs and images concurrently for improved performance
- **Smart Caching**: Caches extracted text to avoid reprocessing unchanged PDFs
- **Image Deduplication**: Prevents processing duplicate images within PDFs
- **Vector Search**: Uses FAISS for efficient similarity search
- **Local LLM Integration**: Leverages Ollama for private, local question-answering
- **Optimized Performance**: Configurable workers, image preprocessing, and reduced context windows

## Requirements

- Python 3.8+
- Tesseract OCR installed on your system
- Ollama with a model installed (e.g., Mistral)

## Installation

1. Install Tesseract OCR:

   - **macOS**: `brew install tesseract`
   - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
   - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

2. Install Ollama and pull a model:

   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull mistral
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
test_rag/
├── main.py              # Main application code
├── requirements.txt     # Python dependencies
├── my_pdfs/            # Place your PDF files here
├── vector_index/       # FAISS index storage (auto-generated)
└── .pdf_cache/         # Cache for extracted text (auto-generated)
```

## Usage

### Basic Query

```bash
python main.py -q "What is the main topic discussed in the documents?"
```

### Rebuild Index

If you've added new PDFs or want to rebuild the vector index:

```bash
python main.py --rebuild -q "Your question here"
```

## Configuration

Edit the configuration variables in [main.py](main.py) (lines 26-33):

```python
DATA_DIR = Path("./my_pdfs")        # PDF directory
INDEX_DIR = Path("./vector_index")  # FAISS index location
CACHE_DIR = Path("./.pdf_cache")    # Cache directory
TOP_K = 3                           # Number of relevant chunks to retrieve
OLLAMA_MODEL = "mistral"            # Ollama model name
MAX_WORKERS = 4                     # Parallel processing workers
OCR_IMAGE_MAX_SIZE = 2000          # Max image dimension for OCR
```

## How It Works

1. **PDF Loading**: The system loads PDFs from the `my_pdfs/` directory
2. **Text Extraction**: Extracts both regular text and performs OCR on images
3. **Caching**: Stores extracted text in `.pdf_cache/` for faster subsequent runs
4. **Chunking**: Splits documents into smaller chunks with overlap
5. **Embedding**: Creates vector embeddings using HuggingFace's MiniLM model
6. **Indexing**: Stores embeddings in a FAISS index for fast retrieval
7. **Query**: Takes your question, finds relevant chunks, and generates answers using Ollama

## Performance Features

- **Parallel Processing**: Processes multiple PDFs and images simultaneously
- **Smart Caching**: Only reprocesses PDFs when they've been modified
- **Image Optimization**: Resizes and converts images to grayscale before OCR
- **Image Deduplication**: Skips duplicate images using MD5 hashing
- **Optimized LLM Settings**: Uses reduced context windows for faster inference

## Dependencies

Key libraries:

- **LangChain**: Framework for LLM applications
- **FAISS**: Vector similarity search
- **PyMuPDF**: PDF processing
- **Pytesseract**: OCR engine
- **HuggingFace**: Embedding models
- **Ollama**: Local LLM integration

See [requirements.txt](requirements.txt) for the complete list.

## License

This project is provided as-is for educational and personal use.
