"""
rag.py
======
Pipeline RAG: recupera chunks relevantes do ChromaDB e gera resposta via Groq.
Embeddings via chromadb.utils.embedding_functions (onnxruntime, sem torch).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

CHROMA_DIR  = Path(__file__).resolve().parent.parent / ".chroma"
GROQ_MODEL  = "llama-3.1-8b-instant"
N_RESULTS   = 5

SYSTEM_PROMPT = """You are a helpful scientific assistant specializing in solar physics \
and space weather. You answer questions about solar flares, coronal mass ejections (CMEs), \
magnetic active regions, and related forecasting methods.

Use ONLY the context provided below to answer. If the answer is not in the context, \
say so clearly — do not invent information.

Answer in the same language as the question (Portuguese or English).

Context:
{context}
"""

HUMAN_PROMPT = "{question}"


class ChromaOnnxEmbeddings(Embeddings):
    """
    Wrapper LangChain-compatível em volta da DefaultEmbeddingFunction do ChromaDB.
    Usa all-MiniLM-L6-v2 via onnxruntime — sem torch, sem HuggingFace runtime.
    """

    def __init__(self):
        self._fn = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._fn(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._fn([text])[0]


def build_chain():
    """Constrói e retorna a chain RAG (chamada uma vez na inicialização do app)."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Defina a variável de ambiente GROQ_API_KEY.")

    if not CHROMA_DIR.exists():
        raise FileNotFoundError(
            f"Índice ChromaDB não encontrado em {CHROMA_DIR}. "
            "Execute `python src/ingest.py` antes de iniciar o app."
        )

    embeddings = ChromaOnnxEmbeddings()

    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": N_RESULTS},
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,
        groq_api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human",  HUMAN_PROMPT),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'doc')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def retrieve_sources(question: str, vectorstore: Chroma) -> list[str]:
    """Retorna lista de fontes usadas na resposta (para exibir no chat)."""
    docs = vectorstore.similarity_search(question, k=N_RESULTS)
    seen, sources = set(), []
    for doc in docs:
        src = doc.metadata.get("source", "")
        name = Path(src).name if src else "desconhecido"
        if name not in seen:
            seen.add(name)
            sources.append(name)
    return sources
