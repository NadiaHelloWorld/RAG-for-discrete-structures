from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .documents import Chunk


def open_collection(path: Path, name: str):
    try:
        import chromadb
    except ImportError as error:
        raise RuntimeError(
            "Chưa cài ChromaDB. Hãy chạy: python -m pip install -r requirements.txt"
        ) from error

    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    return client.get_or_create_collection(
        name=name,
        embedding_function=None,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(collection, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        embeddings=[list(vector) for vector in embeddings],
    )
    return len(chunks)


def search_collection(collection, query_vector: Sequence[float], top_k: int) -> list[dict]:
    result = collection.query(
        query_embeddings=[list(query_vector)],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    return [
        {
            "text": document,
            "metadata": metadata or {},
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]

