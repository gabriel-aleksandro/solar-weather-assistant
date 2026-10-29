"""
rag.py
======
Pipeline RAG: recupera chunks relevantes do ChromaDB e gera resposta via Groq.
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

CHROMA_DIR      = Path(__file__).resolve().parent.parent / ".chroma"
# Usa a Inference API do HF — sem carregar modelo em RAM
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
N_RESULTS       = 5   # chunks recuperados por consulta

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


def build_chain():
    """Constrói e retorna a chain RAG (chamada uma vez na inicialização do app)."""
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise RuntimeError("Defina a variável de ambiente GROQ_API_KEY.")

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("Defina a variável de ambiente HF_TOKEN.")

    embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name=EMBEDDING_MODEL,
    )

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
