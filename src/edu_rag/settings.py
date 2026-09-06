from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COURSE_SOURCE_ROOT = PROJECT_ROOT / "data" / "source" / "cau_truc_roi_rac"
CHROMA_PATH = PROJECT_ROOT / "data" / "index" / "chroma_db"

COURSE_ID = "cau_truc_roi_rac"
COLLECTION_NAME = os.getenv("EDU_RAG_COLLECTION", "cau_truc_roi_rac_qwen3")
EMBEDDING_MODEL = os.getenv("EDU_RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")

