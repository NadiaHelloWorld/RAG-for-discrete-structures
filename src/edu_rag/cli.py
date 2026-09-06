from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .documents import Chunk, load_chunks
from .embedding import QwenEmbedder
from .generation import LocalQwenAnswerer
from .settings import (
    CHROMA_PATH,
    COLLECTION_NAME,
    COURSE_ID,
    COURSE_SOURCE_ROOT,
    EMBEDDING_MODEL,
    PROJECT_ROOT,
    VISUAL_COLLECTION_NAME,
    VISUAL_EMBEDDING_MODEL,
)
from .store import open_collection, search_collection, upsert_chunks
from .visual import QwenVLEmbedder, discover_visual_documents, render_pages


def _build_embedder() -> QwenEmbedder:
    print(f"Đang tải embedding model: {EMBEDDING_MODEL}")
    return QwenEmbedder(EMBEDDING_MODEL, device="cpu")


def _best_visual_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _build_visual_embedder(model_name: str, device: str) -> QwenVLEmbedder:
    selected_device = _best_visual_device() if device == "auto" else device
    print(f"Đang tải visual embedding model: {model_name}")
    print(f"Device visual embedding: {selected_device}")
    return QwenVLEmbedder(model_name, device=selected_device)


def _ingest(source_root: Path, collection_name: str) -> None:
    documents = sorted(source_root.rglob("*"))
    print(f"Đang đọc tài liệu trong: {source_root}")
    chunks = load_chunks(source_root)
    if not chunks:
        raise SystemExit("Không tìm thấy DOCX/PDF/PPTX để ingest.")
    print(f"Đã tạo {len(chunks)} chunks từ {len(documents)} file/thư mục được quét.")

    embedder = _build_embedder()
    embeddings = embedder.encode_documents([chunk.text for chunk in chunks])
    collection = open_collection(CHROMA_PATH, collection_name)
    count = upsert_chunks(collection, chunks, embeddings)
    print(f"Đã lưu {count} chunks vào ChromaDB: {CHROMA_PATH}")
    print(f"Collection: {collection_name} | Tổng số bản ghi: {collection.count()}")


def _retrieve(question: str, top_k: int, collection_name: str) -> list[dict]:
    embedder = _build_embedder()
    collection = open_collection(CHROMA_PATH, collection_name)
    if collection.count() == 0:
        raise SystemExit("ChromaDB đang trống. Hãy chạy lệnh ingest trước.")
    return search_collection(collection, embedder.encode_query(question), top_k)


def _visual_ingest(
    source_root: Path,
    collection_name: str,
    model_name: str,
    device: str,
    batch_size: int,
    dpi: int,
    render_dir: Path,
) -> None:
    source_files = discover_visual_documents(source_root)
    if not source_files:
        raise SystemExit("Không tìm thấy PDF/PPTX để tạo visual index.")

    visual_pages = []
    for source_file in source_files:
        print(f"Đang render: {source_file.name}")
        pages = render_pages(source_file, render_dir, dpi=dpi)
        visual_pages.extend(pages)
        print(f"  Đã tạo {len(pages)} ảnh trang/slide.")

    if not visual_pages:
        raise SystemExit("Không tạo được ảnh từ PDF/PPTX.")

    embedder = _build_visual_embedder(model_name, device)
    vectors = embedder.encode_images(
        [page.image_path for page in visual_pages],
        batch_size=batch_size,
    )

    chunks: list[Chunk] = []
    for page in visual_pages:
        try:
            relative_source = page.source_file.relative_to(PROJECT_ROOT)
        except ValueError:
            relative_source = Path(page.source_file.name)
        digest = hashlib.sha1(
            f"{relative_source}:{page.page_number}".encode("utf-8")
        ).hexdigest()[:16]
        lesson_id = next(
            (part for part in relative_source.parts if part.startswith("buoi_")),
            "unknown",
        )
        metadata = {
            "course_id": COURSE_ID,
            "lesson_id": lesson_id,
            "source_file": str(relative_source),
            "file_name": page.source_file.name,
            "file_type": page.source_file.suffix.lower().lstrip("."),
            "resource_type": "visual_page",
            "page_number": page.page_number,
            "asset_path": str(page.image_path),
        }
        chunks.append(
            Chunk(
                chunk_id=f"{COURSE_ID}-visual-{digest}",
                text=(
                    f"Hình ảnh trang/slide {page.page_number} "
                    f"của {page.source_file.name}"
                ),
                metadata=metadata,
            )
        )

    collection = open_collection(CHROMA_PATH, collection_name)
    count = upsert_chunks(collection, chunks, vectors)
    print(f"Đã lưu {count} visual vectors vào ChromaDB: {CHROMA_PATH}")
    print(f"Collection: {collection_name} | Tổng số bản ghi: {collection.count()}")


def _visual_search(
    question: str,
    top_k: int,
    collection_name: str,
    model_name: str,
    device: str,
) -> None:
    embedder = _build_visual_embedder(model_name, device)
    collection = open_collection(CHROMA_PATH, collection_name)
    if collection.count() == 0:
        raise SystemExit("Visual ChromaDB đang trống. Hãy chạy lệnh ingest-visual trước.")

    results = search_collection(collection, embedder.encode_query(question), top_k)
    print(f"\nCâu hỏi visual: {question}")
    print(f"Tìm thấy {len(results)} slide/trang liên quan:\n")
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"[{index}] {metadata.get('file_name', 'unknown')} | "
            f"trang/slide {metadata.get('page_number', '?')} | "
            f"distance={result['distance']:.4f}"
        )
        print(f"Ảnh đã render: {metadata.get('asset_path', 'unknown')}\n")


def _print_results(question: str, results: list[dict]) -> None:
    print(f"\nCâu hỏi: {question}")
    print(f"Tìm thấy {len(results)} bằng chứng:\n")
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"[{index}] {metadata.get('file_name', 'unknown')} | "
            f"{metadata.get('lesson_id', 'unknown')} | "
            f"chunk {metadata.get('chunk_index', '?')} | "
            f"distance={result['distance']:.4f}"
        )
        print(result["text"][:1400].strip())
        print()


def _answer(
    question: str,
    top_k: int,
    collection_name: str,
    generation_model: str,
    max_new_tokens: int,
) -> None:
    results = _retrieve(question, top_k, collection_name)
    if not results:
        print("Không tìm thấy bằng chứng phù hợp trong tài liệu môn học.")
        return

    answerer = LocalQwenAnswerer(
        model_name=generation_model,
        max_new_tokens=max_new_tokens,
    )
    print(f"\nTrợ lý: {answerer.answer(question, results)}\n")
    print("Nguồn:")
    for index, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(
            f"[{index}] {metadata.get('file_name', 'unknown')} | "
            f"{metadata.get('lesson_id', 'unknown')} | "
            f"chunk {metadata.get('chunk_index', '?')}"
        )


def _chat(
    top_k: int,
    collection_name: str,
    generation_model: str,
    max_new_tokens: int,
) -> None:
    embedder = _build_embedder()
    collection = open_collection(CHROMA_PATH, collection_name)
    if collection.count() == 0:
        raise SystemExit("ChromaDB đang trống. Hãy chạy lệnh ingest trước.")

    answerer = LocalQwenAnswerer(
        model_name=generation_model,
        max_new_tokens=max_new_tokens,
    )
    print("Đã sẵn sàng. Nhập câu hỏi; gõ 'exit' hoặc 'thoát' để kết thúc.")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nĐã kết thúc.")
            return
        if question.casefold() in {"exit", "quit", "thoát"}:
            print("Đã kết thúc.")
            return
        if not question:
            continue
        results = search_collection(collection, embedder.encode_query(question), top_k)
        if not results:
            print("Trợ lý: Không tìm thấy bằng chứng phù hợp trong tài liệu môn học.")
            continue
        print(f"\nTrợ lý: {answerer.answer(question, results)}\n")
        print("Nguồn:")
        for index, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(
                f"[{index}] {metadata.get('file_name', 'unknown')} | "
                f"{metadata.get('lesson_id', 'unknown')} | "
                f"chunk {metadata.get('chunk_index', '?')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG môn Cấu trúc rời rạc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Đọc tài liệu và tạo Chroma index")
    ingest_parser.add_argument("--source-root", type=Path, default=COURSE_SOURCE_ROOT)
    ingest_parser.add_argument("--collection", default=COLLECTION_NAME)

    visual_ingest_parser = subparsers.add_parser(
        "ingest-visual", help="Render PDF/PPTX và tạo visual Chroma index"
    )
    visual_ingest_parser.add_argument("--source-root", type=Path, default=COURSE_SOURCE_ROOT)
    visual_ingest_parser.add_argument("--collection", default=VISUAL_COLLECTION_NAME)
    visual_ingest_parser.add_argument("--model", default=VISUAL_EMBEDDING_MODEL)
    visual_ingest_parser.add_argument(
        "--device", choices=["auto", "cpu", "mps"], default="auto"
    )
    visual_ingest_parser.add_argument("--batch-size", type=int, default=2)
    visual_ingest_parser.add_argument("--dpi", type=int, default=120)
    visual_ingest_parser.add_argument(
        "--render-dir", type=Path, default=CHROMA_PATH / "visual_assets"
    )

    visual_search_parser = subparsers.add_parser(
        "visual-search", help="Tìm slide/trang bằng câu hỏi tự nhiên"
    )
    visual_search_parser.add_argument("question")
    visual_search_parser.add_argument("--top-k", type=int, default=5)
    visual_search_parser.add_argument("--collection", default=VISUAL_COLLECTION_NAME)
    visual_search_parser.add_argument("--model", default=VISUAL_EMBEDDING_MODEL)
    visual_search_parser.add_argument(
        "--device", choices=["auto", "cpu", "mps"], default="auto"
    )

    search_parser = subparsers.add_parser("search", help="Chỉ truy xuất bằng chứng")
    search_parser.add_argument("question")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--collection", default=COLLECTION_NAME)

    ask_parser = subparsers.add_parser("ask", help="Trả lời một câu hỏi bằng Qwen local")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--collection", default=COLLECTION_NAME)
    ask_parser.add_argument("--generation-model", default="Qwen/Qwen3-1.7B")
    ask_parser.add_argument("--max-new-tokens", type=int, default=384)

    chat_parser = subparsers.add_parser("chat", help="Hỏi đáp liên tục bằng câu hỏi tự nhiên")
    chat_parser.add_argument("--top-k", type=int, default=5)
    chat_parser.add_argument("--collection", default=COLLECTION_NAME)
    chat_parser.add_argument("--generation-model", default="Qwen/Qwen3-1.7B")
    chat_parser.add_argument("--max-new-tokens", type=int, default=384)

    args = parser.parse_args()
    if args.command == "ingest":
        _ingest(args.source_root, args.collection)
    elif args.command == "ingest-visual":
        _visual_ingest(
            args.source_root,
            args.collection,
            args.model,
            args.device,
            args.batch_size,
            args.dpi,
            args.render_dir,
        )
    elif args.command == "visual-search":
        _visual_search(
            args.question,
            args.top_k,
            args.collection,
            args.model,
            args.device,
        )
    elif args.command == "search":
        _print_results(args.question, _retrieve(args.question, args.top_k, args.collection))
    elif args.command == "ask":
        _answer(
            args.question,
            args.top_k,
            args.collection,
            args.generation_model,
            args.max_new_tokens,
        )
    else:
        _chat(
            args.top_k,
            args.collection,
            args.generation_model,
            args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
