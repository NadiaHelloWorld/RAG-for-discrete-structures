from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


VISUAL_EXTENSIONS = {".pdf", ".pptx"}


@dataclass(frozen=True)
class VisualPage:
    source_file: Path
    page_number: int
    image_path: Path


def discover_visual_documents(source_root: Path) -> list[Path]:
    return sorted(
        path
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VISUAL_EXTENSIONS
    )


def _find_binary(name: str, env_name: str, candidates: tuple[Path, ...]) -> str:
    configured = os.getenv(env_name)
    if configured:
        return configured
    on_path = shutil.which(name)
    if on_path:
        return on_path
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        f"Không tìm thấy {name}. Hãy cài công cụ cần thiết hoặc đặt biến {env_name}."
    )


def _soffice() -> str:
    return _find_binary(
        "soffice",
        "EDU_RAG_SOFFICE",
        (
            Path("/opt/homebrew/bin/soffice"),
            Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
        ),
    )


def _pdftoppm() -> str:
    return _find_binary(
        "pdftoppm",
        "EDU_RAG_PDFTOPPM",
        (Path("/opt/homebrew/bin/pdftoppm"), Path("/usr/local/bin/pdftoppm")),
    )


def _page_number(path: Path) -> int:
    match = re.search(r"-(\d+)\.png$", path.name)
    return int(match.group(1)) if match else 0


def render_pages(source_file: Path, output_root: Path, dpi: int = 120) -> list[VisualPage]:
    """Render PDF pages or PowerPoint slides into deterministic PNG files."""
    source_file = source_file.resolve()
    output_dir = output_root / source_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="edu-rag-render-") as temp_dir:
        temp_path = Path(temp_dir)
        if source_file.suffix.lower() == ".pptx":
            converted_pdf = temp_path / f"{source_file.stem}.pdf"
            libreoffice_profile = temp_path / "libreoffice-profile"
            subprocess.run(
                [
                    _soffice(),
                    f"-env:UserInstallation=file://{libreoffice_profile}",
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(temp_path),
                    str(source_file),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if not converted_pdf.exists():
                raise RuntimeError(f"LibreOffice không tạo được PDF từ {source_file}")
            pdf_file = converted_pdf
        elif source_file.suffix.lower() == ".pdf":
            pdf_file = source_file
        else:
            raise ValueError(f"Định dạng visual chưa hỗ trợ: {source_file.suffix}")

        subprocess.run(
            [
                _pdftoppm(),
                "-png",
                "-r",
                str(dpi),
                str(pdf_file),
                str(output_dir / "page"),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    images = sorted(output_dir.glob("page-*.png"), key=_page_number)
    return [VisualPage(source_file, _page_number(image), image) for image in images]


class QwenVLEmbedder:
    """Qwen3-VL adapter for text queries and local slide/page images."""

    def __init__(self, model_name: str, device: str = "mps") -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=True,
        )

    @staticmethod
    def _encode_kwargs(batch_size: int) -> dict:
        return {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": True,
        }

    def encode_images(self, image_paths: list[Path], batch_size: int = 2) -> list[list[float]]:
        inputs = [{"image": str(path.resolve())} for path in image_paths]
        vectors = self.model.encode(
            inputs,
            prompt="Represent this course slide or document page for visual retrieval.",
            **self._encode_kwargs(batch_size),
        )
        return vectors.tolist()

    def encode_query(self, question: str) -> list[float]:
        vector = self.model.encode(
            [question],
            prompt="Find the course slides or document pages relevant to the user's question.",
            **self._encode_kwargs(1),
        )[0]
        return vector.tolist()
