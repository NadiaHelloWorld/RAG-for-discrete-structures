from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from .settings import COURSE_ID, PROJECT_ROOT


SUPPORTED_EXTENSIONS = {".docx", ".pdf", ".pptx"}


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    metadata: dict[str, str | int]


def discover_documents(source_root: Path) -> list[Path]:
    candidates = sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    # When the same lesson is supplied as both PDF and PPTX, keep one text
    # representation so the vector index does not contain duplicate chunks.
    selected: dict[tuple[Path, str], Path] = {}
    priority = {".docx": 0, ".pdf": 1, ".pptx": 2}
    for path in candidates:
        key = (path.parent, path.stem.casefold())
        current = selected.get(key)
        if current is None or priority[path.suffix.lower()] < priority[current.suffix.lower()]:
            selected[key] = path
    return sorted(selected.values())


def _convert_to_markdown(path: Path) -> str:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as error:
        raise RuntimeError(
            "Chưa cài Docling. Hãy dùng Python 3.10+ và chạy: "
            "python -m pip install -r requirements.txt"
        ) from error

    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


def _split_markdown(markdown: str, max_chars: int = 1600) -> list[str]:
    paragraphs = [part.strip() for part in markdown.split("\n\n") if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_size = 0

    for paragraph in paragraphs:
        if current and current_size + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            # Retain the last paragraph as lightweight overlap for context.
            current = current[-1:]
            current_size = len(current[0]) if current else 0
        current.append(paragraph)
        current_size += len(paragraph) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _metadata(path: Path, chunk_index: int) -> dict[str, str | int]:
    relative = path.relative_to(PROJECT_ROOT)
    parts = relative.parts
    lesson_id = next((part for part in parts if part.startswith("buoi_")), "unknown")
    return {
        "course_id": COURSE_ID,
        "lesson_id": lesson_id,
        "source_file": str(relative),
        "file_name": path.name,
        "file_type": path.suffix.lower().lstrip("."),
        "resource_type": "exercise" if "bai_tap" in path.stem.lower() else "document",
        "chunk_index": chunk_index,
    }


def load_chunks(source_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in discover_documents(source_root):
        markdown = _convert_to_markdown(path)
        for index, text in enumerate(_split_markdown(markdown)):
            metadata = _metadata(path, index)
            digest = hashlib.sha1(
                f"{metadata['source_file']}:{index}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(Chunk(chunk_id=f"{COURSE_ID}-{digest}", text=text, metadata=metadata))
    return chunks
