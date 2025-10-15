"""QA chain setup and LLM configuration"""

import warnings

from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import PGVector

from .config import OLLAMA_MODEL, LLM_TEMPERATURE, LLM_NUM_CTX, LLM_NUM_PREDICT, TOP_K

warnings.filterwarnings("ignore")


def get_llm(streaming=False, callbacks=None):
    """Get configured Ollama LLM instance"""
    return Ollama(
        model=OLLAMA_MODEL,
        temperature=LLM_TEMPERATURE,
        num_ctx=LLM_NUM_CTX,
        num_predict=LLM_NUM_PREDICT,
        streaming=streaming,
        callbacks=callbacks,
    )


def make_qa_chain(vs: PGVector, streaming=False, callbacks=None):
    """Create QA chain from vector store"""
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    llm = get_llm(streaming=streaming, callbacks=callbacks)
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type="stuff",
    )
