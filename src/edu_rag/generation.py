from __future__ import annotations

from collections.abc import Sequence


class LocalQwenAnswerer:
    """Generate a grounded answer with a local Qwen3 causal language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-1.7B",
        device: str | None = None,
        max_new_tokens: int = 384,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        dtype = torch.float16 if device == "mps" else torch.float32

        print(f"Đang tải model sinh câu trả lời: {model_name} ({device})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        )
        self.model.to(device)
        self.model.eval()

    @staticmethod
    def _context(results: Sequence[dict]) -> str:
        parts: list[str] = []
        for index, result in enumerate(results, start=1):
            metadata = result.get("metadata", {})
            source = metadata.get("file_name", "unknown")
            lesson = metadata.get("lesson_id", "unknown")
            chunk = metadata.get("chunk_index", "?")
            parts.append(
                f"[{index}] Nguồn: {source} | {lesson} | chunk {chunk}\n"
                f"{result.get('text', '').strip()}"
            )
        return "\n\n".join(parts)

    def answer(self, question: str, results: Sequence[dict]) -> str:
        if not results:
            return "Mình không tìm thấy thông tin phù hợp trong tài liệu môn học."

        context = self._context(results)
        messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý học tập cho môn Cấu trúc rời rạc. "
                    "Hãy trả lời bằng tiếng Việt, chỉ dùng thông tin trong CONTEXT. "
                    "Nếu context không đủ, nói rõ là tài liệu hiện tại chưa đủ thông tin. "
                    "Khi dùng một đoạn nguồn, ghi [1], [2] tương ứng với số nguồn. "
                    "Với bài toán, hãy trình bày các bước ngắn gọn và không tự bịa đề bài."
                ),
            },
            {
                "role": "user",
                "content": f"CONTEXT:\n{context}\n\nCÂU HỎI:\n{question}",
            },
        ]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        inputs = inputs.to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

