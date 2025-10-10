"""QA chain setup and LLM configuration"""
import warnings
warnings.filterwarnings("ignore")

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector

from .config import (
    OLLAMA_MODEL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
    LLM_NUM_PREDICT,
    TOP_K
)


def get_llm():
    """Get configured Ollama LLM instance"""
    return Ollama(
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
    )


def make_qa_chain(vs: PGVector):
    """Create QA chain from vector store"""
    retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    llm = get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
