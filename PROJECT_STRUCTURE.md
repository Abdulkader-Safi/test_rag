# Project Structure

This PDF RAG (Retrieval-Augmented Generation) system has been organized into modular components for better maintainability.

## Directory Structure

```
test_rag/
├── main.py                 # Main entry point and CLI interface
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── pdf_loader.py      # PDF loading and OCR extraction
│   ├── vector_store.py    # Vector store operations
│   └── qa_chain.py        # QA chain and LLM setup
├── my_pdfs/               # Directory for PDF documents
├── .pdf_cache/            # Cache for extracted text
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables (DATABASE_URL)
```

## Module Overview

### `main.py`

- CLI argument parsing
- Interactive and single-query modes
- Coordinates all modules
- Handles user interface with Rich library

### `src/config.py`

- All configuration settings in one place
- Environment variables loading
- Model parameters
- Directory paths
- Performance settings

### `src/pdf_loader.py`

- PDF text extraction using PyPDFLoader
- OCR processing for images in PDFs
- Parallel processing for performance
- Caching system for faster reloading
- Functions:
  - `load_pdfs()` - Main function to load all PDFs
  - `extract_text_from_images()` - OCR processing
  - `_load_single_pdf()` - Load individual PDF with caching

### `src/vector_store.py`

- PostgreSQL vector store operations
- Document chunking
- Embeddings management
- Functions:
  - `ensure_index()` - Connect to vector store
  - `add_pdfs_to_index()` - Add PDFs to database
  - `get_embeddings()` - Get embedding model (singleton)
  - `chunk_docs()` - Split documents into chunks

### `src/qa_chain.py`

- LLM configuration (Ollama)
- QA chain setup
- Retrieval configuration
- Functions:
  - `get_llm()` - Get configured LLM instance
  - `make_qa_chain()` - Create QA chain from vector store

## Usage

### Single Query Mode

```bash
python main.py -q "your question here"
```

### Interactive Mode

```bash
python main.py
```

### Add PDFs to Database

```bash
python main.py --add-pdfs
```

### Clear Database

```bash
python main.py --clear
```

## Configuration

Edit `src/config.py` to modify:

- Model settings (LLM, embeddings)
- Performance parameters (workers, cache size)
- Chunk sizes and overlap
- Database settings

## Benefits of This Structure

1. **Separation of Concerns**: Each module has a specific responsibility
2. **Easy Testing**: Individual modules can be tested independently
3. **Maintainability**: Changes to one component don't affect others
4. **Scalability**: Easy to add new features or swap implementations
5. **Configuration Management**: All settings in one centralized location
6. **Reusability**: Modules can be imported and used in other projects
