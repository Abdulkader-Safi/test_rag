# PDF RAG System

A powerful Retrieval-Augmented Generation (RAG) system for querying PDF documents using local LLMs via Ollama.

![PDF RAG SYSTEM](https://abdulkadersafi.com/storage/158/01K6E35JVPR2MCC56CRHNWRNDP.png)

> Read Blog Article [Here](https://abdulkadersafi.com/blog/retrieval-augmented-generation-rag-with-local-pdfs-and-ollama-a-developers-guide)

## Features

- 📄 **PDF Processing**: Extracts text from PDFs including OCR for images
- 🔍 **Vector Search**: PostgreSQL-based vector storage with pgvector
- 🤖 **Local LLM**: Uses Ollama (Mistral) for privacy and cost-effectiveness
- ⚡ **Performance**: Parallel processing and intelligent caching
- 💬 **Two Modes**: Single-query mode or interactive chat
- 🎨 **Rich UI**: Beautiful terminal output with syntax highlighting
- 🏗️ **Modular Design**: Clean, organized code structure for easy maintenance

## Quick Start

### Prerequisites

1. **PostgreSQL with pgvector**

   ```bash
   # macOS
   brew install postgresql pgvector

   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   # Then install pgvector from source
   ```

2. **Ollama**

   ```bash
   # Install Ollama from https://ollama.ai
   brew install ollama  # macOS

   # Pull Mistral model
   ollama pull mistral
   ```

3. **Tesseract OCR**

   ```bash
   # macOS
   brew install tesseract

   # Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   ```

4. **Python 3.10+**

### Installation

1. Clone and setup:

   ```bash
   cd test_rag
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Configure environment:

   ```bash
   cp .env.example .env
   # Edit .env and set your DATABASE_URL
   ```

3. Add your PDFs:

   ```bash
   # Place PDF files in the my_pdfs/ directory
   cp your-documents.pdf my_pdfs/
   ```

4. Index your PDFs:

   ```bash
   python main.py --add-pdfs
   ```

### Usage

**Single Query Mode** (returns only the answer):

```bash
python main.py -q "What are the main topics in these documents?****"
```

**Interactive Mode** (with source documents):

```bash
python main.py
> What specifications are mentioned for tiles?
> exit
```

**Clear Database**:

```bash
python main.py --clear
```

## Project Structure

```folder
test_rag/
├── main.py                 # CLI entry point
├── src/                    # Source code modules
│   ├── __init__.py        # Package initialization
│   ├── config.py          # Configuration settings
│   ├── pdf_loader.py      # PDF extraction & OCR
│   ├── vector_store.py    # Vector database ops
│   └── qa_chain.py        # LLM & QA chain
├── my_pdfs/               # Your PDF documents
├── .pdf_cache/            # Extracted text cache
├── requirements.txt       # Python dependencies
└── .env                   # Environment variables
```

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed documentation.

## Configuration

All settings can be customized in [src/config.py](src/config.py):

```python
# Model Configuration
OLLAMA_MODEL = "mistral"  # Change to "llama2", "phi", etc.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Performance
MAX_WORKERS = 4  # Parallel processing threads
TOP_K = 3  # Number of documents to retrieve

# Text Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

## How It Works

1. **PDF Processing**: Extracts text and runs OCR on images
2. **Chunking**: Splits documents into searchable chunks
3. **Embedding**: Converts text to vectors using HuggingFace embeddings
4. **Storage**: Stores vectors in PostgreSQL with pgvector
5. **Retrieval**: Finds most relevant chunks for your query
6. **Generation**: Sends context to Ollama for answer generation

## Performance Tips

- **Caching**: PDFs are cached after first processing
- **Parallel Processing**: Adjust `MAX_WORKERS` in config
- **Chunk Size**: Smaller chunks = more precise, larger = more context
- **Top K**: More documents = better context but slower

## Troubleshooting

**No results returned?**

- Ensure PDFs are indexed: `python main.py --add-pdfs`
- Check DATABASE_URL in .env
- Verify Ollama is running: `ollama list`

**Slow performance?**

- Reduce `TOP_K` in config
- Increase `MAX_WORKERS` for more parallelization
- Check if PDFs are cached (should see "[CACHE HIT]")

**Out of memory?**

- Reduce `LLM_NUM_CTX` in config
- Process fewer PDFs at once
- Reduce `MAX_WORKERS`

## Advanced Usage

### Using Different Models

Edit `src/config.py`:

```python
OLLAMA_MODEL = "llama2"  # or "phi", "codellama", etc.
```

Then pull the model:

```bash
ollama pull llama2
```

### Custom Embeddings

Edit `src/config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

### Database Migration

To change metadata to JSONB (recommended):

```sql
ALTER TABLE langchain_pg_embedding
ALTER COLUMN cmetadata TYPE JSONB USING cmetadata::JSONB;
```

## Dependencies

- **LangChain**: RAG framework
- **Ollama**: Local LLM inference
- **PostgreSQL + pgvector**: Vector database
- **PyMuPDF**: PDF text extraction
- **Pytesseract**: OCR for images
- **HuggingFace**: Embeddings model
- **Rich**: Terminal UI

## License

This project is provided as-is for educational and personal use.

## Contributing

Contributions welcome! Please check existing issues or create a new one.
