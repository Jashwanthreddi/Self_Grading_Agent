from __future__ import annotations

from app.config import get_settings
from app.decision.confidence import NO_ANSWER, decide
from app.generation.generator import GeminiGenerator
from app.retrieval.retriever import HybridRetriever
from app.verification.verifier import NliVerifier


class SelfGradingAgent:
    """
    End-to-end self-grading question-answering agent.

    Pipeline:

        Question
            ↓
        Hybrid Retrieval
            ↓
        Evidence Passages
            ↓
        Gemini Generation
            ↓
        Draft Answer + Atomic Claims
            ↓
        Independent NLI Verification
            ↓
        Deterministic Confidence Decision
            ↓
        Final Result
    """

    def __init__(self):
        # -------------------------------------------------------------
        # Load application configuration
        # -------------------------------------------------------------

        settings = get_settings()

        self.settings = settings

        # -------------------------------------------------------------
        # Hybrid Retrieval
        # -------------------------------------------------------------
        #
        # HybridRetriever expects its configuration values explicitly.
        # We therefore pass the settings fields individually instead of
        # passing the Settings object itself.
        #

        self.retriever = HybridRetriever(
            faiss_index_path=settings.faiss_index_path,
            bm25_index_path=settings.bm25_index_path,
            embedding_model=settings.embedding_model,
            dense_top_k=settings.dense_top_k,
            bm25_top_k=settings.bm25_top_k,
            final_top_k=settings.final_top_k,
            dense_weight=settings.dense_weight,
            bm25_weight=settings.bm25_weight,
        )

        # -------------------------------------------------------------
        # Generation
        # -------------------------------------------------------------

        self.generator = GeminiGenerator(
            settings
        )

        # -------------------------------------------------------------
        # Independent Verification
        # -------------------------------------------------------------

        self.verifier = NliVerifier(
            settings
        )

    # -----------------------------------------------------------------
    # MAIN PIPELINE
    # -----------------------------------------------------------------

    def run(
        self,
        question: str,
    ) -> dict:
        """
        Run the complete self-grading pipeline.

        Returns:
            dict containing:

            - answer
            - confidence
            - reason
            - evidence_ids
            - passages
            - verification
        """

        # -------------------------------------------------------------
        # Validate question
        # -------------------------------------------------------------

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        # -------------------------------------------------------------
        # STEP 1 — RETRIEVAL
        # -------------------------------------------------------------

        passages = self.retriever.retrieve(
            question
        )

        # -------------------------------------------------------------
        # STEP 2 — GENERATION
        # -------------------------------------------------------------

        draft = self.generator.generate(
            question,
            passages,
        )

        # -------------------------------------------------------------
        # STEP 3 — INDEPENDENT VERIFICATION
        # -------------------------------------------------------------

        verification = self.verifier.verify(
            draft,
            passages,
        )

        # -------------------------------------------------------------
        # STEP 4 — CONFIDENCE DECISION
        # -------------------------------------------------------------

        confidence, reason = decide(
            verification,
            draft,
            self.settings.high_confidence_threshold,
        )

        # -------------------------------------------------------------
        # STEP 5 — FINAL ANSWER
        # -------------------------------------------------------------

        answer = draft.answer.strip()

        # If the confidence engine determines that the evidence
        # is insufficient, explicitly abstain.
        if confidence == NO_ANSWER:
            answer = NO_ANSWER

        # -------------------------------------------------------------
        # RETURN COMPLETE RESULT
        # -------------------------------------------------------------

        return {
            "question": question,
            "answer": answer,
            "confidence": confidence,
            "reason": reason,

            "evidence_ids": [
                passage.passage_id
                for passage in passages
            ],

            "passages": passages,

            "verification": verification,

            "draft": draft,
        }