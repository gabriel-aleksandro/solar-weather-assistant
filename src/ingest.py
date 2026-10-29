"""
ingest.py
=========
Indexa documentos da pasta docs/ no ChromaDB (persistente em .chroma/).
Embeddings via chromadb.utils.embedding_functions (onnxruntime, sem torch).

Uso:
    python src/ingest.py

Suporta: PDF, Markdown (.md), texto (.txt)
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

DOCS_DIR   = Path(__file__).resolve().parent.parent / "docs"
CHROMA_DIR = Path(__file__).resolve().parent.parent / ".chroma"

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100


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


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.rglob("*")):
        if path.suffix == ".pdf":
            print(f"  Carregando PDF: {path.name}")
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif path.suffix in (".md", ".txt"):
            print(f"  Carregando texto: {path.name}")
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
    return docs


def ingest():
    print(f"[1/3] Carregando documentos de {DOCS_DIR} ...")
    raw_docs = load_documents()
    print(f"  -> {len(raw_docs)} páginas/seções carregadas.")

    print("[2/3] Dividindo em chunks ...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    print(f"  -> {len(chunks)} chunks gerados.")

    print("[3/3] Gerando embeddings e salvando no ChromaDB ...")
    embeddings = ChromaOnnxEmbeddings()

    # Remove índice antigo se existir (evita conflito de dimensões)
    import shutil
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"  (índice antigo removido)")

    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print(f"  -> {len(chunks)} chunks indexados em {CHROMA_DIR}")
    print("Indexação concluída.")


if __name__ == "__main__":
    ingest()
