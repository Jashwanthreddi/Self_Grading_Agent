import json
import re
import time
from pathlib import Path

from app.pipeline.agent import SelfGradingAgent


# ----------------------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------------------

# IMPORTANT:
# The benchmark should not repeatedly retry TPM rate-limit errors.
# A retry does not help when the requested completion would still exceed
# the organization's tokens-per-minute limit.
EVALUATION_RETRY_ATTEMPTS = 1

# Short fallback delay for transient non-rate-limit errors.
EVALUATION_RETRY_BASE_DELAY = 2

# Delay between successful questions.
#
# This is intentionally short. The evaluator should not spend several
# minutes sleeping between every question. Rate-limit errors are handled
# separately and recorded as execution failures.
EVALUATION_QUESTION_DELAY = 3


# ----------------------------------------------------------------------
# Required confidence labels
# ----------------------------------------------------------------------

HIGH_CONFIDENCE = (
    "High confidence — the answer is clearly supported by the sources."
)

LOW_CONFIDENCE = (
    "Low confidence — the available evidence is incomplete, ambiguous, "
    "or requires an unsupported inference."
)

NO_ANSWER = (
    "I don't know — the documents do not contain enough evidence to answer."
)


# ----------------------------------------------------------------------
# Abstention detection
# ----------------------------------------------------------------------

ABSTENTION_PHRASES = (
    "i don't know",
    "i do not know",
    "not enough evidence",
    "not provided",
    "does not provide",
    "not specified",
    "cannot determine",
    "can't determine",
    "cannot answer",
    "can't answer",
    "insufficient evidence",
    "not available",
    "unknown",
)


class EvaluationRunner:
    """
    Run the benchmark through the self-grading agent.

    Evaluation is deliberately separated into two concerns:

    1. Agent execution:
       retrieval -> generation -> verification -> confidence

    2. Benchmark scoring:
       compare the returned answer/confidence against manually
       curated reference facts.

    API/execution failures are recorded separately and are excluded
    from model-quality denominators.
    """

    def __init__(self, agent: SelfGradingAgent):
        self.agent = agent

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    @staticmethod
    def load_questions(path: Path) -> list[dict]:
        """Load and validate the benchmark dataset."""

        with path.open("r", encoding="utf-8") as file:
            questions = json.load(file)

        if not isinstance(questions, list):
            raise ValueError("Evaluation dataset must contain a JSON list.")

        if not questions:
            raise ValueError("Evaluation dataset is empty.")

        required_fields = {
            "id",
            "category",
            "question",
            "reference_answer",
            "expected_confidence",
        }

        for item in questions:
            missing = required_fields - item.keys()

            if missing:
                raise ValueError(
                    f"Question {item.get('id', '<unknown>')} "
                    f"is missing fields: {sorted(missing)}"
                )

        return questions

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for deterministic benchmark comparison."""

        text = str(text).lower()

        text = text.replace("—", "-")
        text = text.replace("–", "-")

        text = re.sub(
            r"[^a-z0-9%$./+-]+",
            " ",
            text,
        )

        return " ".join(text.split())

    @classmethod
    def _contains(
        cls,
        answer: str,
        phrase: str,
    ) -> bool:
        """Check whether a factual anchor occurs in the answer."""

        answer_normalized = cls._normalize(answer)
        phrase_normalized = cls._normalize(phrase)

        if not phrase_normalized:
            return False

        return phrase_normalized in answer_normalized

    @classmethod
    def _is_abstention(cls, answer: str) -> bool:
        """Determine whether the answer clearly abstains."""

        normalized = cls._normalize(answer)

        return any(
            phrase in normalized
            for phrase in ABSTENTION_PHRASES
        )

    # ------------------------------------------------------------------
    # Reference facts
    # ------------------------------------------------------------------

    @staticmethod
    def _reference_facts(item: dict) -> list[str]:
        """
        Return manually curated factual anchors.

        required_facts is preferred because a reference answer can
        contain multiple facts that all need to be present.
        """

        facts = item.get("required_facts")

        if facts:
            return [
                str(fact)
                for fact in facts
                if str(fact).strip()
            ]

        return [str(item["reference_answer"])]

    @staticmethod
    def _forbidden_facts(item: dict) -> list[str]:
        """
        Return facts that should not appear in the answer.

        This is especially important for trap questions.
        """

        facts = item.get("forbidden_facts", [])

        return [
            str(fact)
            for fact in facts
            if str(fact).strip()
        ]

    # ------------------------------------------------------------------
    # Evidence serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_evidence(result: dict) -> list[dict]:
        """
        Convert RetrievedPassage objects into JSON/CSV-safe dictionaries.
        """

        evidence = result.get("evidence", [])

        serialized = []

        for passage in evidence:
            if hasattr(passage, "model_dump"):
                data = passage.model_dump()
            elif isinstance(passage, dict):
                data = passage
            else:
                data = {
                    "document_id": getattr(
                        passage,
                        "document_id",
                        "",
                    ),
                    "document_name": getattr(
                        passage,
                        "document_name",
                        "",
                    ),
                    "content": getattr(
                        passage,
                        "content",
                        "",
                    ),
                    "score": getattr(
                        passage,
                        "score",
                        0.0,
                    ),
                    "retrieval_method": getattr(
                        passage,
                        "retrieval_method",
                        "",
                    ),
                }

            serialized.append(
                {
                    "document_id": data.get("document_id", ""),
                    "document_name": data.get("document_name", ""),
                    "content": data.get("content", ""),
                    "score": data.get("score", 0.0),
                    "retrieval_method": data.get(
                        "retrieval_method",
                        "",
                    ),
                }
            )

        return serialized

    # ------------------------------------------------------------------
    # Verification serialization
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_verification(result: dict) -> dict:
        """Convert verifier output into a JSON-safe dictionary."""

        verification = result.get("verification", {})

        if hasattr(verification, "model_dump"):
            return verification.model_dump()

        if isinstance(verification, dict):
            return verification

        return {}

    # ------------------------------------------------------------------
    # Correctness
    # ------------------------------------------------------------------

    @classmethod
    def _answer_is_correct(
        cls,
        result: dict,
        item: dict,
    ) -> tuple[bool, str]:
        """
        Determine factual correctness independently from confidence.

        Confidence and correctness are intentionally separate.

        A response can:
        - be factually correct but overconfident
        - be factually correct but over-cautious
        - be factually incorrect and overconfident
        - correctly abstain
        """

        answer = str(
            result.get("answer", "")
        ).strip()

        confidence = result.get("confidence")

        category = item["category"]

        if not answer:
            return (
                False,
                "The application returned an empty answer.",
            )

        # --------------------------------------------------------------
        # Unanswerable
        # --------------------------------------------------------------

        if category == "unanswerable":

            if confidence != NO_ANSWER:
                return (
                    False,
                    "The question is unanswerable, but the application "
                    "did not use the required I don't know confidence label.",
                )

            if not cls._is_abstention(answer):
                return (
                    False,
                    "The application selected I don't know but did not "
                    "clearly abstain in the answer text.",
                )

            return (
                True,
                "Correct abstention: the requested fact is absent from "
                "the knowledge base.",
            )

        # --------------------------------------------------------------
        # Partially supported / ambiguous
        # --------------------------------------------------------------

        if category == "partially_supported_ambiguous":

            # The assignment expects Low confidence for this category.
            # Factual correctness is still checked separately below.
            verification = result.get("verification", {})

            expected_low = confidence == LOW_CONFIDENCE

            # For ambiguous questions, the response should either
            # explicitly communicate uncertainty or provide the correct
            # qualified answer.
            reference_facts = cls._reference_facts(item)

            facts_present = all(
                cls._contains(answer, fact)
                for fact in reference_facts
            )

            forbidden_present = any(
                cls._contains(answer, fact)
                for fact in cls._forbidden_facts(item)
            )

            if expected_low and not forbidden_present:
                if facts_present or cls._is_abstention(answer):
                    return (
                        True,
                        "Correct handling of partially supported or "
                        "ambiguous evidence.",
                    )

                # Some partial questions intentionally expect a qualified
                # negative/uncertain response rather than the full fact.
                if verification.get("supported") is False:
                    return (
                        True,
                        "The verifier correctly rejected an "
                        "over-strong interpretation of the evidence.",
                    )

            return (
                False,
                "The response did not correctly handle the ambiguous "
                "or conditional evidence.",
            )

        # --------------------------------------------------------------
        # Answerable + trap
        # --------------------------------------------------------------

        reference_facts = cls._reference_facts(item)
        forbidden_facts = cls._forbidden_facts(item)

        missing_facts = [
            fact
            for fact in reference_facts
            if not cls._contains(answer, fact)
        ]

        present_forbidden = [
            fact
            for fact in forbidden_facts
            if cls._contains(answer, fact)
        ]

        if missing_facts:
            return (
                False,
                "The answer is missing required factual anchor(s): "
                + ", ".join(missing_facts),
            )

        if present_forbidden:
            return (
                False,
                "The answer contains forbidden/contradictory "
                "factual anchor(s): "
                + ", ".join(present_forbidden),
            )

        return (
            True,
            "The answer contains all required factual anchors and "
            "does not contain forbidden facts.",
        )

    # ------------------------------------------------------------------
    # Confidence correctness
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_is_correct(
        result: dict,
        item: dict,
    ) -> bool:
        """Compare predicted confidence with the curated expectation."""

        return result.get("confidence") == item["expected_confidence"]

    # ------------------------------------------------------------------
    # Dangerous overconfidence
    # ------------------------------------------------------------------

    @classmethod
    def _is_dangerous_overconfidence(
        cls,
        result: dict,
        item: dict,
        answer_correct: bool,
    ) -> bool:
        """
        Detect a high-confidence prediction that is factually wrong.

        This is the key reliability failure for a self-grading system.
        """

        return (
            result.get("confidence") == HIGH_CONFIDENCE
            and not answer_correct
        )

    # ------------------------------------------------------------------
    # Over-cautious abstention
    # ------------------------------------------------------------------

    @classmethod
    def _is_over_cautious_abstention(
        cls,
        result: dict,
        item: dict,
        answer_correct: bool,
    ) -> bool:
        """
        Detect unnecessary I don't know predictions on answerable
        questions where the generated answer was factually correct.
        """

        return (
            item["category"] in {
                "answerable",
                "trap",
            }
            and result.get("confidence") == NO_ANSWER
            and answer_correct
        )

    # ------------------------------------------------------------------
    # Rate-limit handling
    # ------------------------------------------------------------------

    @staticmethod
    def _is_rate_limit_error(error: Exception) -> bool:
        """
        Detect Groq/OpenAI-style rate-limit errors without depending on
        a specific exception class.
        """

        text = str(error).lower()

        return (
            "ratelimiterror" in text
            or "rate limit" in text
            or "rate_limit_exceeded" in text
            or "tokens per minute" in text
            or "tokens per day" in text
            or "429" in text
        )

    # ------------------------------------------------------------------
    # Single question
    # ------------------------------------------------------------------

    def evaluate_question(
        self,
        item: dict,
    ) -> dict:
        """
        Execute one benchmark question and return a standardized result.
        """

        question_id = item["id"]
        question_text = item["question"]

        base_result = {
            "question_id": question_id,
            "category": item["category"],
            "question": question_text,
            "reference_answer": item["reference_answer"],
            "expected_confidence": item["expected_confidence"],
            "predicted_answer": "",
            "predicted_confidence": "",
            "correct": False,
            "confidence_correct": False,
            "dangerous_overconfidence": False,
            "over_cautious_abstention": False,
            "correctness_notes": "",
            "retrieved_evidence": [],
            "verification": {},
            "execution_error": "",
        }

        try:
            raw_result = self.agent.run(question_text)

            # ----------------------------------------------------------
            # Normalize agent result
            # ----------------------------------------------------------

            answer = str(
                raw_result.get("answer", "")
            ).strip()

            confidence = raw_result.get(
                "confidence",
                "",
            )

            normalized_result = {
                "answer": answer,
                "confidence": confidence,
                "evidence": raw_result.get(
                    "evidence",
                    [],
                ),
                "verification": raw_result.get(
                    "verification",
                    {},
                ),
            }

            # ----------------------------------------------------------
            # Independent scoring
            # ----------------------------------------------------------

            answer_correct, correctness_notes = (
                self._answer_is_correct(
                    normalized_result,
                    item,
                )
            )

            confidence_correct = (
                self._confidence_is_correct(
                    normalized_result,
                    item,
                )
            )

            dangerous = (
                self._is_dangerous_overconfidence(
                    normalized_result,
                    item,
                    answer_correct,
                )
            )

            over_cautious = (
                self._is_over_cautious_abstention(
                    normalized_result,
                    item,
                    answer_correct,
                )
            )

            base_result.update(
                {
                    "predicted_answer": answer,
                    "predicted_confidence": confidence,
                    "correct": answer_correct,
                    "confidence_correct": confidence_correct,
                    "dangerous_overconfidence": dangerous,
                    "over_cautious_abstention": over_cautious,
                    "correctness_notes": correctness_notes,
                    "retrieved_evidence": self._serialize_evidence(
                        normalized_result
                    ),
                    "verification": self._serialize_verification(
                        normalized_result
                    ),
                }
            )

            return base_result

        except Exception as error:

            error_text = (
                f"{type(error).__name__}: {error}"
            )

            base_result["execution_error"] = error_text

            if self._is_rate_limit_error(error):
                base_result["correctness_notes"] = (
                    "Execution failed because the LLM provider "
                    "rate limit was reached. This question is excluded "
                    "from model-quality denominators."
                )
            else:
                base_result["correctness_notes"] = (
                    "Execution failed before the question could be "
                    "evaluated. This question is excluded from "
                    "model-quality denominators."
                )

            return base_result

    # ------------------------------------------------------------------
    # Full benchmark
    # ------------------------------------------------------------------

    def run(
        self,
        questions: list[dict],
    ) -> list[dict]:
        """
        Run the complete benchmark.

        Every question gets a result entry, including execution failures.
        This makes the benchmark auditable and reproducible.
        """

        results = []

        total = len(questions)

        for index, item in enumerate(
            questions,
            start=1,
        ):
            print(
                f"[{index}/{total}] "
                f"{item['id']}: "
                f"{item['question']}"
            )

            result = self.evaluate_question(item)

            results.append(result)

            if result["execution_error"]:

                print(
                    "  ERROR: "
                    + result["execution_error"]
                )

                # IMPORTANT:
                # Do not retry TPM failures. Repeating the request only
                # consumes more quota and produces another failure.
                if self._is_rate_limit_error(
                    Exception(
                        result["execution_error"]
                    )
                ):
                    print(
                        "  Rate limit detected. "
                        "Skipping retries for this question."
                    )

            else:

                print(
                    "  Confidence: "
                    + result["predicted_confidence"]
                )

                print(
                    "  Correct: "
                    + str(result["correct"])
                )

                if result["dangerous_overconfidence"]:
                    print(
                        "  WARNING: dangerous overconfidence detected."
                    )

            # Small delay between successful questions only.
            #
            # We do not sleep 30 seconds after every failed question,
            # because a TPM limit should be handled by the provider
            # rather than by repeatedly burning time.
            if index < total and not result["execution_error"]:
                print(
                    f"  Waiting {EVALUATION_QUESTION_DELAY}s..."
                )
                time.sleep(
                    EVALUATION_QUESTION_DELAY
                )

        return results

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    @staticmethod
    def save_results(
        results: list[dict],
        output_path: Path,
    ) -> None:
        """Save complete benchmark results as JSON."""

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                results,
                file,
                indent=2,
                ensure_ascii=False,
            )
