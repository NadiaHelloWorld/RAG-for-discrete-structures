from __future__ import annotations

import argparse
from pathlib import Path

from .documents import load_chunks
from .embedding import QwenEmbedder
from .settings import CHROMA_PATH, COLLECTION_NAME, COURSE_SOURCE_ROOT, EMBEDDING_MODEL
from .store import open_collection, search_collection, upsert_chunks


def _build_embedder() -> QwenEmbedder:
    print(f"Đang tải embedding model: {EMBEDDING_MODEL}")
    return QwenEmbedder(EMBEDDING_MODEL, device="cpu")


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


def _chat(top_k: int, collection_name: str) -> None:
    embedder = _build_embedder()
    collection = open_collection(CHROMA_PATH, collection_name)
    if collection.count() == 0:
        raise SystemExit("ChromaDB đang trống. Hãy chạy lệnh ingest trước.")

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
        _print_results(question, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG môn Cấu trúc rời rạc")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Đọc tài liệu và tạo Chroma index")
    ingest_parser.add_argument("--source-root", type=Path, default=COURSE_SOURCE_ROOT)
    ingest_parser.add_argument("--collection", default=COLLECTION_NAME)

    ask_parser = subparsers.add_parser("ask", help="Truy xuất bằng chứng cho một câu hỏi")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--collection", default=COLLECTION_NAME)

    chat_parser = subparsers.add_parser("chat", help="Hỏi đáp liên tục bằng câu hỏi tự nhiên")
    chat_parser.add_argument("--top-k", type=int, default=5)
    chat_parser.add_argument("--collection", default=COLLECTION_NAME)

    args = parser.parse_args()
    if args.command == "ingest":
        _ingest(args.source_root, args.collection)
    elif args.command == "ask":
        _print_results(args.question, _retrieve(args.question, args.top_k, args.collection))
    else:
        _chat(args.top_k, args.collection)


if __name__ == "__main__":
    main()
