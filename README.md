# RAG for Discrete Structures

Trợ lý học tập hỏi đáp theo tài liệu cho môn **Cấu trúc rời rạc**.

Project bắt đầu với tài liệu Buổi 3: *Bài tập Chương 3*, gồm quan hệ hai ngôi,
quan hệ tương đương và quan hệ thứ tự.

## Kiến trúc

```text
DOCX/PDF
   ↓
Docling: đọc cấu trúc, tiêu đề, bảng, công thức và hình ảnh
   ↓
Chunk có thông tin nguồn
   ↓
Qwen/Qwen3-Embedding-0.6B: tạo vector văn bản
   ↓
ChromaDB: lưu vector trên máy
   ↓
Người học nhập câu hỏi → truy xuất bằng chứng → trả lời có nguồn
```

## Tài nguyên hiện có

```text
data/source/cau_truc_roi_rac/buoi_3/bai_tap_chuong_3.docx
```

## Cài đặt

Project dùng Python 3.10 hoặc mới hơn vì Docling hiện không còn hỗ trợ Python
3.9 ở các phiên bản mới. Sau khi tạo môi trường ảo:

```bash
python -m pip install -r requirements.txt
```

## Chạy thử

Đọc tài liệu và tạo Chroma index:

```bash
PYTHONPATH=src python -m edu_rag.cli ingest
```

Hỏi trực tiếp một câu:

```bash
PYTHONPATH=src python -m edu_rag.cli ask "Quan hệ tương đương là gì?"
```

Mở chế độ hỏi đáp liên tục:

```bash
PYTHONPATH=src python -m edu_rag.cli chat
```

Người dùng chỉ cần nhập câu hỏi; gõ `exit` để kết thúc.

## Dữ liệu sinh tự động

Thư mục `data/index/chroma_db/`, model cache và file tạm không được commit lên
GitHub. Chúng sẽ được tạo lại bằng lệnh `ingest` trên máy mới.

## Trạng thái

- [x] Tách project riêng cho môn Cấu trúc rời rạc.
- [x] Thêm tài liệu Buổi 3.
- [x] Chọn Docling, Qwen3 Embedding và ChromaDB làm kiến trúc mục tiêu.
- [ ] Cài môi trường Python 3.10/3.11 và chạy ingest thật.
- [ ] Bổ sung video/slide các buổi tiếp theo.
- [ ] Thêm lớp sinh câu trả lời LLM sau khi retrieval ổn định.
