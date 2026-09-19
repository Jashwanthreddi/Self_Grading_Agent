from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    groq_api_key: str | None = None

    llm_model: str = "openai/gpt-oss-120b"
    llm_max_tokens: int = 512
    llm_timeout: float = 30.0
    llm_max_retries: int = 0

    embedding_model: str = "BAAI/bge-base-en-v1.5"

    dense_top_k: int = 5
    bm25_top_k: int = 5
    final_top_k: int = 3

    dense_weight: float = 0.6
    bm25_weight: float = 0.4

    knowledge_base_path: Path = Path(
        "data/knowledge_base"
    )

    evaluation_path: Path = Path(
        "data/evaluation/test_questions.json"
    )

    faiss_index_path: Path = Path(
        "indexes/faiss"
    )

    bm25_index_path: Path = Path(
        "indexes/bm25"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()

    # Local development: .env / environment variable
    if settings.groq_api_key:
        return settings

    # Streamlit deployment: st.secrets
    try:
        import streamlit as st

        secret_key = st.secrets.get(
            "GROQ_API_KEY"
        )

        if secret_key:
            settings.groq_api_key = str(
                secret_key
            )

    except Exception:
        pass

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set it in .env or Streamlit secrets."
        )

    return settings