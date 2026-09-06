# RAG for Discrete Structures

Trợ lý học tập hỏi đáp theo tài liệu cho môn **Cấu trúc rời rạc**.

Project hiện có tài liệu Buổi 1 về *Cơ sở logic* và Buổi 3 về *Quan hệ*, gồm
mệnh đề, phép toán logic, quan hệ hai ngôi, quan hệ tương đương và quan hệ thứ tự.

## Kiến trúc

```text
DOCX/PDF/PPTX
   ↓
Docling: đọc cấu trúc, tiêu đề, bảng, công thức và hình ảnh
   ↓
 ┌────────────────────────────────┬─────────────────────────────────┐
 │ Nội dung chữ                  │ Trang/slide dạng hình ảnh        │
 │ Qwen3-Embedding-0.6B          │ Qwen3-VL-Embedding-2B            │
 │ Chroma collection text        │ Chroma collection visual         │
 └────────────────────────────────┴─────────────────────────────────┘
                         ↓
        Câu hỏi tự nhiên → truy xuất text + visual
                         ↓
                 Qwen local trả lời có nguồn
```

## Tài nguyên hiện có

```text
data/source/cau_truc_roi_rac/buoi_3/bai_tap_chuong_3.docx
data/source/cau_truc_roi_rac/buoi_3/Chuong_3_Quan_he.pptx
data/source/cau_truc_roi_rac/buoi_1/Chuong_1_Co_so_logic_slides_bai_tap.pptx
data/source/cau_truc_roi_rac/buoi_1/Chuong_1_Co_so_logic_slides_bai_tap.pdf
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

Tạo visual index cho PDF/PPTX. Lệnh này tự chuyển từng trang/slide thành PNG,
sau đó dùng Qwen3-VL-Embedding-2B để lưu vector hình ảnh vào collection riêng:

```bash
PYTHONPATH=src python -m edu_rag.cli ingest-visual
```

Trên Mac có MPS, mặc định lệnh dùng MPS; có thể ép chạy CPU nếu thiếu bộ nhớ:

```bash
PYTHONPATH=src python -m edu_rag.cli ingest-visual --device cpu --batch-size 1
```

Kiểm tra truy xuất slide bằng câu hỏi tự nhiên:

```bash
PYTHONPATH=src python -m edu_rag.cli visual-search \
  "Slide nào giải thích quan hệ tương đương?"
```

Hỏi trực tiếp về nội dung nhìn thấy trong slide/trang. Lệnh này truy xuất cả
collection chữ và collection hình ảnh, sau đó đưa ảnh liên quan cho
Qwen3-VL-2B-Instruct đọc và trả lời:

```bash
PYTHONPATH=src python -m edu_rag.cli ask-multimodal \
  "Hãy giải thích hình minh họa về quan hệ tương đương trong slide."
```

Nếu máy thiếu bộ nhớ, chạy bản an toàn hơn trên CPU:

```bash
PYTHONPATH=src python -m edu_rag.cli ask-multimodal \
  --device cpu --top-k 3 \
  "Hãy giải thích hình minh họa về quan hệ tương đương trong slide."
```

Visual embedding giúp tìm đúng slide/trang dựa trên nội dung hình ảnh và câu hỏi.
Qwen3-VL-2B-Instruct dùng ở bước sau để đọc ảnh và viết câu trả lời.
Nếu PDF và PPTX có cùng tên trong một buổi học, index sẽ chọn một bản để tránh
lưu trùng nội dung; cả hai file gốc vẫn được giữ trong repo.

Nếu muốn dùng chung ChromaDB với notebook trong thư mục `rag_test`, đặt biến
môi trường trước khi chạy:

```bash
export EDU_RAG_CHROMA_PATH="/Users/VoThiXuanHoa/Downloads/rag_test/chroma_db"
PYTHONPATH=src python -m edu_rag.cli ingest
```

Hỏi trực tiếp một câu bằng Qwen local:

```bash
export EDU_RAG_CHROMA_PATH="/Users/VoThiXuanHoa/Downloads/rag_test/chroma_db"
PYTHONPATH=src python -m edu_rag.cli ask "Quan hệ tương đương là gì?"
```

Mở chế độ hỏi đáp liên tục:

```bash
PYTHONPATH=src python -m edu_rag.cli chat
```

Người dùng chỉ cần nhập câu hỏi; gõ `exit` để kết thúc.

`ask` và `chat` dùng `Qwen/Qwen3-1.7B` chạy local trên MPS hoặc CPU để sinh
câu trả lời. Lần đầu chạy, model sẽ được tải về Hugging Face cache trên máy.
Có thể chọn bản lớn hơn khi máy đủ bộ nhớ:

```bash
PYTHONPATH=src python -m edu_rag.cli ask \
  --generation-model Qwen/Qwen3-4B \
  "Quan hệ tương đương là gì?"
```

Lệnh `search` chỉ hiển thị bằng chứng, không tải model sinh câu trả lời:

```bash
PYTHONPATH=src python -m edu_rag.cli search "Quan hệ tương đương là gì?"
```

## Dữ liệu sinh tự động

Thư mục `data/index/chroma_db/`, model cache và file tạm không được commit lên
GitHub. Chúng sẽ được tạo lại bằng lệnh `ingest` trên máy mới.

## Trạng thái

- [x] Tách project riêng cho môn Cấu trúc rời rạc.
- [x] Thêm tài liệu Buổi 1 và Buổi 3.
- [x] Chọn Docling, Qwen3 Embedding và ChromaDB làm kiến trúc mục tiêu.
- [x] Cài môi trường Python 3.10/3.11 và chạy ingest thật.
- [ ] Bổ sung video/slide các buổi tiếp theo.
- [x] Thêm lớp sinh câu trả lời Qwen local và giao diện `ask`/`chat`.
- [x] Thêm pipeline render PDF/PPTX và visual retrieval bằng Qwen3-VL-Embedding-2B.
- [x] Nối Qwen3-VL-2B-Instruct để trả lời dựa trên ảnh slide/trang.
- [ ] Đánh giá chất lượng câu trả lời trên bộ câu hỏi có đáp án.
