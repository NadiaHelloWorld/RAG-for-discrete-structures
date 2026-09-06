from __future__ import annotations

from collections.abc import Sequence


class QwenEmbedder:
    """Small adapter around Qwen3 Embedding via SentenceTransformers."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if hasattr(self.model, "encode_document"):
            vectors = self.model.encode_document(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        else:
            vectors = self.model.encode(
                list(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=True,
            )
        return vectors.tolist()

    def encode_query(self, query: str) -> list[float]:
        if hasattr(self.model, "encode_query"):
            vector = self.model.encode_query(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
        else:
            vector = self.model.encode(
                [query],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )[0]
        return vector.tolist()

