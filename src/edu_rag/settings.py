from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_SOURCE_ROOT = PROJECT_ROOT / "data" / "source" / "cau_truc_roi_rac"
DEFAULT_CHROMA_PATH = PROJECT_ROOT / "data" / "index" / "chroma_db"
CHROMA_PATH = Path(os.getenv("EDU_RAG_CHROMA_PATH", str(DEFAULT_CHROMA_PATH)))

COURSE_ID = "cau_truc_roi_rac"
COLLECTION_NAME = os.getenv("EDU_RAG_COLLECTION", "cau_truc_roi_rac_qwen3")
VISUAL_COLLECTION_NAME = os.getenv(
    "EDU_RAG_VISUAL_COLLECTION", "cau_truc_roi_rac_visual_qwen3vl"
)
EMBEDDING_MODEL = os.getenv("EDU_RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
VISUAL_EMBEDDING_MODEL = os.getenv(
    "EDU_RAG_VISUAL_EMBEDDING_MODEL", "Qwen/Qwen3-VL-Embedding-2B"
)
VISUAL_GENERATION_MODEL = os.getenv(
    "EDU_RAG_VISUAL_GENERATION_MODEL", "Qwen/Qwen3-VL-2B-Instruct"
)
