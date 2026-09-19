from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):

    # ============================================================
    # Hosted generation model
    # ============================================================

    gemini_api_key: str | None = None

    generation_model: str = "gemini-3.8-flash"

    # ============================================================
    # Local models
    # ============================================================

    embedding_model: str = "BAAI/bge-base-en-v1.5"

    nli_model: str = "cross-encoder/nli-MiniLM2-L6-H768"

    # ============================================================
    # Paths
    # ============================================================

    knowledge_base_path: Path = (
        PROJECT_ROOT / "data" / "knowledge_base"
    )

    evaluation_path: Path = (
        PROJECT_ROOT
        / "data"
        / "evaluation"
        / "test_questions.json"
    )

    faiss_index_path: Path = (
        PROJECT_ROOT / "indexes" / "faiss"
    )

    bm25_index_path: Path = (
        PROJECT_ROOT / "indexes" / "bm25"
    )

    metadata_path: Path = (
        PROJECT_ROOT / "indexes" / "metadata.json"
    )

    # ============================================================
    # Chunking
    # ============================================================

    chunk_size: int = 900

    chunk_overlap: int = 120

    # ============================================================
    # Retrieval
    # ============================================================

    dense_top_k: int = 6

    bm25_top_k: int = 6

    final_top_k: int = 4

    dense_weight: float = 0.60

    bm25_weight: float = 0.40

    # ============================================================
    # Verification thresholds
    # ============================================================

    nli_entailment_threshold: float = 0.70

    nli_contradiction_threshold: float = 0.30

    high_confidence_threshold: float = 0.82

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:

    settings = Settings()

    if not settings.gemini_api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to the project .env file."
        )

    return settings