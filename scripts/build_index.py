import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

from app.config import get_settings
from app.ingestion.indexer import build_indexes


def main() -> None:
    settings = get_settings()

    result = build_indexes(
        knowledge_base_path=settings.knowledge_base_path,
        faiss_index_path=settings.faiss_index_path,
        bm25_index_path=settings.bm25_index_path,
        embedding_model_name=settings.embedding_model,
    )

    print("\nIndex build completed.")
    print(f"Documents: {result['documents']}")
    print(f"Chunks: {result['chunks']}")
    print(
        f"Embedding dimension: "
        f"{result['embedding_dimension']}"
    )


if __name__ == "__main__":
    main()