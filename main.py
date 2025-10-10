import io
import os
import sys
import warnings

# Redirect stderr immediately to suppress all warnings
sys.stderr = open(os.devnull, 'w')

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_SUPPRESS_WARNINGS"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress specific deprecation warnings
import logging
logging.getLogger("langchain").setLevel(logging.ERROR)

import argparse
from rich.console import Console
from rich.panel import Panel

from src.vector_store import ensure_index, add_pdfs_to_index
from src.qa_chain import make_qa_chain


def main():
    """Main entry point for the PDF RAG system"""
    parser = argparse.ArgumentParser(
        description="PDF RAG System - Ask questions about your PDF documents"
    )
    parser.add_argument(
        "--add-pdfs",
        action="store_true",
        help="Add PDFs from my_pdfs directory to vector store"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all vectors from the database"
    )
    parser.add_argument(
        "-q", "--query",
        type=str,
        help="Ask a single question and exit"
    )
    args = parser.parse_args()

    console = Console()

    vs = None
    qa = None

    try:
        # Handle clear command
        if args.clear:
            console.print("[bold yellow]Clearing vector store...[/bold yellow]")
            vs = ensure_index()
            vs.delete_collection()
            console.print("[bold green]Vector store cleared successfully.[/bold green]")
            sys.exit(0)

        # Handle add PDFs command
        if args.add_pdfs:
            add_pdfs_to_index()
            sys.exit(0)

        # Initialize QA chain
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

        # Interactive mode
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


if __name__ == "__main__":
    main()
