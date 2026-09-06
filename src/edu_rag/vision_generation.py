from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class LocalQwenVLAnswerer:
    """Answer questions by looking at retrieved local slide/page images."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-2B-Instruct",
        device: str | None = None,
        max_new_tokens: int = 384,
    ) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "mps" else torch.float32

        print(f"Đang tải model Qwen-VL trả lời: {model_name} ({device})")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _valid_visual_results(results: Sequence[dict], max_images: int) -> list[dict]:
        valid: list[dict] = []
        for result in results:
            image_path = Path(result.get("metadata", {}).get("asset_path", ""))
            if image_path.exists():
                valid.append(result)
            if len(valid) >= max_images:
                break
        return valid

    @staticmethod
    def _text_context(results: Sequence[dict]) -> str:
        parts: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            parts.append(
                f"[T{index}] {metadata.get('file_name', 'unknown')} | "
                f"{metadata.get('lesson_id', 'unknown')} | "
                f"chunk {metadata.get('chunk_index', '?')}\n"
                f"{result.get('text', '').strip()}"
            )
        return "\n\n".join(parts)

    def answer(
        self,
        question: str,
        visual_results: Sequence[dict],
        text_results: Sequence[dict] = (),
        max_images: int = 3,
    ) -> str:
        valid_visuals = self._valid_visual_results(visual_results, max_images)
        if not valid_visuals and not text_results:
            return "Mình không tìm thấy bằng chứng phù hợp trong tài liệu môn học."

        content: list[dict] = []
        for index, result in enumerate(valid_visuals, start=1):
            metadata = result.get("metadata", {})
            content.append(
                {
                    "type": "image",
                    "image": str(Path(metadata["asset_path"]).resolve()),
                }
            )
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Ảnh V{index}: {metadata.get('file_name', 'unknown')}, "
                        f"trang/slide {metadata.get('page_number', '?')}"
                    ),
                }
            )

        text_context = self._text_context(text_results)
        prompt = (
            "Bạn là trợ lý học tập môn Cấu trúc rời rạc. Hãy trả lời bằng tiếng Việt. "
            "Quan sát các ảnh slide/trang được cung cấp và dùng thêm TEXT CONTEXT nếu có. "
            "Chỉ kết luận những gì đọc được từ ảnh hoặc context; nếu ảnh mờ/thiếu thông tin "
            "hãy nói rõ. Ghi nguồn ảnh bằng [V1], [V2] và nguồn chữ bằng [T1], [T2].\n\n"
            f"TEXT CONTEXT:\n{text_context or '(không có)'}\n\n"
            f"CÂU HỎI: {question}"
        )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as error:
            raise RuntimeError(
                "Chưa cài qwen-vl-utils. Hãy chạy: python -m pip install -r requirements.txt"
            ) from error

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
