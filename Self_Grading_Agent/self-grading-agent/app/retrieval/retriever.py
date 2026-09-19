from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.models.schemas import RetrievedPassage


class HybridRetriever:
    """Retrieve evidence using dense and BM25 search."""

    def __init__(
        self,
        faiss_index_path: Path,
        bm25_index_path: Path,
        embedding_model: str,
        dense_top_k: int,
        bm25_top_k: int,
        final_top_k: int,
        dense_weight: float,
        bm25_weight: float,
    ):
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.final_top_k = final_top_k
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight

        self.faiss_index = faiss.read_index(
            str(faiss_index_path / "index.faiss")
        )

        with (faiss_index_path / "chunks.json").open(
            "r",
            encoding="utf-8",
        ) as file:
            self.chunks = json.load(file)

        with (bm25_index_path / "index.pkl").open(
            "rb",
        ) as file:
            self.bm25 = pickle.load(file)

        self.embedding_model = SentenceTransformer(
            embedding_model
        )

        if self.faiss_index.ntotal != len(self.chunks):
            raise ValueError(
                "FAISS index and chunk metadata are inconsistent."
            )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def _dense_search(
        self,
        query: str,
    ) -> list[tuple[int, float]]:
        embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )

        embedding = np.asarray(
            embedding,
            dtype="float32",
        )

        scores, indices = self.faiss_index.search(
            embedding,
            min(
                self.dense_top_k,
                len(self.chunks),
            ),
        )

        return [
            (int(index), float(score))
            for index, score in zip(
                indices[0],
                scores[0],
            )
            if index >= 0
        ]

    def _bm25_search(
        self,
        query: str,
    ) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(
            self._tokenize(query)
        )

        ranked_indices = np.argsort(scores)[::-1]

        return [
            (
                int(index),
                float(scores[index]),
            )
            for index in ranked_indices[
                :self.bm25_top_k
            ]
        ]

    @staticmethod
    def _normalize_scores(
        results: list[tuple[int, float]],
    ) -> dict[int, float]:
        if not results:
            return {}

        values = [
            score
            for _, score in results
        ]

        minimum = min(values)
        maximum = max(values)

        if maximum == minimum:
            return {
                index: 1.0
                for index, _ in results
            }

        return {
            index: (score - minimum)
            / (maximum - minimum)
            for index, score in results
        }

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievedPassage]:
        """
        Return the top evidence passages using
        weighted hybrid retrieval.
        """

        if not query.strip():
            raise ValueError(
                "Query cannot be empty."
            )

        dense_results = self._dense_search(query)
        bm25_results = self._bm25_search(query)

        dense_scores = self._normalize_scores(
            dense_results
        )

        bm25_scores = self._normalize_scores(
            bm25_results
        )

        candidate_ids = (
            set(dense_scores)
            | set(bm25_scores)
        )

        fused = []

        for index in candidate_ids:
            dense_score = dense_scores.get(
                index,
                0.0,
            )

            bm25_score = bm25_scores.get(
                index,
                0.0,
            )

            hybrid_score = (
                self.dense_weight
                * dense_score
                + self.bm25_weight
                * bm25_score
            )

            fused.append(
                (
                    index,
                    dense_score,
                    bm25_score,
                    hybrid_score,
                )
            )

        fused.sort(
            key=lambda item: item[3],
            reverse=True,
        )

        results = []

        for (
            index,
            dense_score,
            bm25_score,
            hybrid_score,
        ) in fused[: self.final_top_k]:

            chunk = self.chunks[index]

            doc_id = chunk.get(
                "doc_id",
                chunk.get(
                    "document_id",
                    "",
                ),
            )

            title = chunk.get(
                "title",
                chunk.get(
                    "document_name",
                    "",
                ),
            )

            source = chunk.get(
                "source",
                chunk.get(
                    "document_name",
                    "",
                ),
            )

            text = chunk.get(
                "text",
                chunk.get(
                    "content",
                    "",
                ),
            )

            passage_id = chunk.get(
                "passage_id",
                f"{doc_id}-chunk-{index}",
            )

            dense_present = index in dense_scores
            bm25_present = index in bm25_scores

            if dense_present and bm25_present:
                method = "hybrid"
            elif dense_present:
                method = "dense"
            else:
                method = "bm25"

            results.append(
                RetrievedPassage(
                    passage_id=passage_id,
                    doc_id=doc_id,
                    title=title,
                    source=source,
                    text=text,
                    dense_score=round(
                        dense_score,
                        4,
                    ),
                    bm25_score=round(
                        bm25_score,
                        4,
                    ),
                    hybrid_score=round(
                        hybrid_score,
                        4,
                    ),
                    retrieval_method=method,
                )
            )

        return results