import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.ingestion.loader import load_documents


MAX_CHUNK_SIZE = 1800


def split_text(
    text: str,
    max_size: int = MAX_CHUNK_SIZE,
) -> list[str]:
    """Split long sections while preserving paragraph boundaries."""

    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_size:
            current = f"{current}\n\n{paragraph}".strip()
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_size:
            current = paragraph
        else:
            for start in range(0, len(paragraph), max_size):
                chunk = paragraph[start:start + max_size].strip()

                if chunk:
                    chunks.append(chunk)

            current = ""

    if current:
        chunks.append(current)

    return chunks


def create_chunks(
    documents: list[dict],
) -> list[dict]:
    """Create metadata-rich chunks from Markdown documents."""

    chunks = []

    for document in documents:
        sections = re.split(
            r"(?=^#{1,3}\s+)",
            document["content"],
            flags=re.MULTILINE,
        )

        chunk_number = 1

        for section in sections:
            section = section.strip()

            if not section:
                continue

            section_chunks = split_text(section)

            for content in section_chunks:
                chunks.append(
                    {
                        "chunk_id": (
                            f"{document['document_id']}-"
                            f"C{chunk_number:03d}"
                        ),
                        "document_id": document["document_id"],
                        "document_name": document["document_name"],
                        "document_type": document["document_type"],
                        "last_updated": document["last_updated"],
                        "content": content,
                    }
                )

                chunk_number += 1

    return chunks


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25 retrieval."""

    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def build_indexes(
    knowledge_base_path: Path,
    faiss_index_path: Path,
    bm25_index_path: Path,
    embedding_model_name: str,
) -> dict:
    """Build and persist FAISS and BM25 indexes."""

    documents = load_documents(knowledge_base_path)
    chunks = create_chunks(documents)

    if not chunks:
        raise ValueError(
            "No chunks were created from the knowledge base."
        )

    faiss_index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    bm25_index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    texts = [
        chunk["content"]
        for chunk in chunks
    ]

    model = SentenceTransformer(
        embedding_model_name
    )

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(
        dimension
    )

    faiss_index.add(embeddings)

    faiss.write_index(
        faiss_index,
        str(
            faiss_index_path / "index.faiss"
        ),
    )

    tokenized_texts = [
        tokenize(text)
        for text in texts
    ]

    bm25 = BM25Okapi(
        tokenized_texts
    )

    with (
        bm25_index_path / "index.pkl"
    ).open("wb") as file:
        pickle.dump(
            bm25,
            file,
        )

    for output_path in (
        faiss_index_path / "chunks.json",
        bm25_index_path / "chunks.json",
    ):
        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                chunks,
                file,
                ensure_ascii=False,
                indent=2,
            )

    return {
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_dimension": dimension,
    }