import os
import sys
import warnings
import logging
import argparse

from rich.console import Console
from rich.panel import Panel
from langchain.callbacks.base import BaseCallbackHandler

from src.vector_store import ensure_index, add_pdfs_to_index
from src.qa_chain import make_qa_chain

# Store original stderr for error output
_original_stderr = sys.stderr

# Temporarily disable stderr redirection for debugging
# sys.stderr = open(os.devnull, "w")

# Suppress all warnings
warnings.filterwarnings("ignore")
os.environ["LANGCHAIN_SUPPRESS_WARNINGS"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"

# Suppress specific deprecation warnings

logging.getLogger("langchain").setLevel(logging.ERROR)


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to console"""

    def __init__(self, console: Console):
        self.console = console

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.console.print(token, end="", style="")


def main():
    """Main entry point for the PDF RAG system"""
    parser = argparse.ArgumentParser(
        description="PDF RAG System - Ask questions about your PDF documents"
    )
    parser.add_argument(
        "--add-pdfs",
        action="store_true",
        help="Add PDFs from my_pdfs directory to vector store",
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

        # Initialize QA chain with streaming
        streaming_handler = StreamingCallbackHandler(console)
        with console.status("[bold green]Initializing QA chain...[/bold green]"):
            vs = ensure_index()
            qa = make_qa_chain(vs, streaming=True, callbacks=[streaming_handler])

        # Handle single query mode
        if args.query:
            console.print("[bold green]Answer:[/bold green] ", end="")
            try:
                result = qa({"query": args.query})
                console.print()
                if not result:
                    console.print("[yellow]No result returned from query[/yellow]")
            except Exception as e:
                sys.stderr = _original_stderr
                console.print(f"\n[bold red]Error during query:[/bold red] {e}")
                import traceback
                traceback.print_exc()
            sys.exit(0)

        # Interactive mode
        console.print(
            "[bold green]Starting interactive chat. Type 'exit' or press Ctrl+C to quit.[/bold green]"
        )

        while True:
            try:
                query = console.input("[bold cyan]> [/bold cyan]")
                if query.lower() == "exit":
                    break

                console.print("[bold green]Answer:[/bold green] ", end="")

                result = qa({"query": query})
                console.print()

                if result.get("source_documents"):
                    console.print("\n[bold]Source Documents:[/bold]")
                    for doc in result["source_documents"]:
                        console.print(
                            Panel(
                                f"[cyan]{doc.metadata['source']}[/cyan] (Page {doc.metadata.get('page', 'N/A')})\n\n{doc.page_content}",
                                title="[bold yellow]Source[/bold yellow]",
                                border_style="yellow",
                            )
                        )

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
