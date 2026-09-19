from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from app.pipeline.agent import SelfGradingAgent


# ----------------------------------------------------------------------
# Evaluation configuration
# ----------------------------------------------------------------------

EVALUATION_RETRY_ATTEMPTS = 3

# Short delay for transient provider failures.
EVALUATION_RETRY_BASE_DELAY = 5

# Small pacing interval between benchmark questions.
EVALUATION_QUESTION_DELAY = 5


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
)


class EvaluationRunner:
    """
    Run the benchmark through the production self-grading agent.

    Correctness is evaluated independently from confidence.

    The benchmark uses manually curated reference answers and
    question categories to determine factual correctness.
    """

    def __init__(
        self,
        agent: SelfGradingAgent,
        questions: list[dict] | None = None,
    ):
        self.agent = agent
        self.questions = questions or []

        # Results are populated after run().
        self.results: list[dict] = []

        # Metrics are populated after run().
        self.metrics = None

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    @staticmethod
    def load_questions(path: Path) -> list[dict]:
        with path.open("r", encoding="utf-8") as file:
            questions = json.load(file)

        if not isinstance(questions, list):
            raise ValueError(
                "Evaluation dataset must contain a JSON list."
            )

        if not questions:
            raise ValueError(
                "Evaluation dataset is empty."
            )

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
    # Text normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """
        Normalize text for deterministic comparison.

        This intentionally avoids fuzzy semantic matching.
        """

        text = str(text or "").lower()

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

        answer_normalized = cls._normalize(answer)
        phrase_normalized = cls._normalize(phrase)

        if not phrase_normalized:
            return False

        return phrase_normalized in answer_normalized

    @classmethod
    def _is_abstention(
        cls,
        answer: str,
    ) -> bool:

        normalized = cls._normalize(answer)

        return any(
            phrase in normalized
            for phrase in ABSTENTION_PHRASES
        )

    # ------------------------------------------------------------------
    # Reference-answer evaluation
    # ------------------------------------------------------------------

    @classmethod
    def _reference_facts(
        cls,
        item: dict,
    ) -> list[str]:

        facts = item.get("required_facts")

        if facts:
            return [
                str(fact)
                for fact in facts
                if str(fact).strip()
            ]

        return [
            str(item["reference_answer"])
        ]

    @classmethod
    def _forbidden_facts(
        cls,
        item: dict,
    ) -> list[str]:

        facts = item.get(
            "forbidden_facts",
            [],
        )

        return [
            str(fact)
            for fact in facts
            if str(fact).strip()
        ]

    # ------------------------------------------------------------------
    # Answer correctness
    # ------------------------------------------------------------------

    @classmethod
    def _answer_is_correct(
        cls,
        result: dict,
        item: dict,
    ) -> tuple[bool, str]:

        answer = result.get(
            "answer",
            "",
        ).strip()

        category = item["category"]

        # --------------------------------------------------------------
        # Basic response validation
        # --------------------------------------------------------------

        if not answer:
            return (
                False,
                "The application returned an empty answer.",
            )

        # --------------------------------------------------------------
        # Unanswerable
        # --------------------------------------------------------------

        if category == "unanswerable":

            correct_confidence = (
                result.get("confidence") == NO_ANSWER
            )

            if not correct_confidence:
                return (
                    False,
                    "The question is unanswerable, but the "
                    "application did not abstain with the required "
                    "\"I don't know\" confidence label.",
                )

            if not cls._is_abstention(answer):
                return (
                    False,
                    "The application selected abstention confidence "
                    "but did not clearly abstain from answering.",
                )

            return (
                True,
                "Correct abstention: the benchmark reference states "
                "that the requested fact is absent from the knowledge base.",
            )

        # --------------------------------------------------------------
        # Partially supported / ambiguous
        # --------------------------------------------------------------

        if category == "partially_supported_ambiguous":

            if result.get("confidence") != LOW_CONFIDENCE:
                return (
                    False,
                    "The question contains incomplete, conditional, "
                    "or ambiguous evidence and therefore requires "
                    "Low confidence.",
                )

            verification = result.get(
                "verification",
                {},
            ) or {}

            if verification.get(
                "contradiction",
                False,
            ):
                return (
                    False,
                    "The generated answer contradicted the retrieved evidence.",
                )

            if cls._is_abstention(answer):
                return (
                    False,
                    "The system was appropriately cautious but failed "
                    "to provide the supported portion of the answer.",
                )

            facts = cls._reference_facts(item)

            matched = [
                fact
                for fact in facts
                if cls._contains(
                    answer,
                    fact,
                )
            ]

            if facts and len(matched) == len(facts):
                return (
                    True,
                    "The answer captures the benchmark's required "
                    "supported facts while remaining Low confidence.",
                )

            keywords = item.get(
                "required_keywords",
                [],
            )

            if keywords:

                normalized_answer = cls._normalize(
                    answer
                )

                missing = [
                    keyword
                    for keyword in keywords
                    if cls._normalize(keyword)
                    not in normalized_answer
                ]

                if not missing:
                    return (
                        True,
                        "The answer contains all manually defined "
                        "factual anchors for the ambiguous case.",
                    )

                return (
                    False,
                    "The answer is Low confidence but misses "
                    f"required factual anchors: {missing}.",
                )

            if (
                verification.get(
                    "supported",
                    False,
                )
                and verification.get(
                    "evidence_sufficient",
                    False,
                )
                and not verification.get(
                    "unsupported_inference",
                    False,
                )
            ):
                return (
                    True,
                    "The answer is consistent with the evidence and "
                    "appropriately uses Low confidence for an ambiguous "
                    "or conditional question.",
                )

            return (
                False,
                "The answer was cautious, but the verifier did not "
                "establish sufficient support for the generated claim.",
            )

        # --------------------------------------------------------------
        # Answerable / trap
        # --------------------------------------------------------------

        if category in {
            "answerable",
            "trap",
        }:

            if result.get("confidence") == NO_ANSWER:
                return (
                    False,
                    "The benchmark contains sufficient evidence, "
                    "but the application abstained.",
                )

            if cls._is_abstention(answer):
                return (
                    False,
                    "The application abstained even though the "
                    "benchmark contains a supported answer.",
                )

            forbidden = cls._forbidden_facts(
                item
            )

            for fact in forbidden:
                if cls._contains(
                    answer,
                    fact,
                ):
                    return (
                        False,
                        "The answer contains a benchmark-defined "
                        f"forbidden fact: {fact!r}.",
                    )

            facts = cls._reference_facts(
                item
            )

            if all(
                cls._contains(
                    answer,
                    fact,
                )
                for fact in facts
            ):
                return (
                    True,
                    "The application answer contains all required "
                    "reference facts.",
                )

            keywords = item.get(
                "required_keywords",
                [],
            )

            if keywords:

                normalized_answer = cls._normalize(
                    answer
                )

                missing = [
                    keyword
                    for keyword in keywords
                    if cls._normalize(keyword)
                    not in normalized_answer
                ]

                if not missing:
                    return (
                        True,
                        "The application answer contains all "
                        "manually defined factual anchors.",
                    )

                return (
                    False,
                    "The answer is missing required factual "
                    f"anchors: {missing}.",
                )

            return (
                False,
                "The answer does not contain the benchmark's "
                "required reference fact.",
            )

        return (
            False,
            f"Unknown benchmark category: {category!r}.",
        )

    # ------------------------------------------------------------------
    # Confidence correctness
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence_is_correct(
        predicted: str,
        expected: str,
    ) -> bool:

        mapping = {
            "High confidence": HIGH_CONFIDENCE,
            "Low confidence": LOW_CONFIDENCE,
            "I don't know": NO_ANSWER,
        }

        return predicted == mapping.get(
            expected,
            "",
        )

    # ------------------------------------------------------------------
    # Calibration risk indicators
    # ------------------------------------------------------------------

    @staticmethod
    def _dangerous_overconfidence(
        answer_correct: bool,
        predicted: str,
    ) -> bool:

        return (
            not answer_correct
            and predicted == HIGH_CONFIDENCE
        )

    @staticmethod
    def _over_cautious_abstention(
        answer_correct: bool,
        expected: str,
        predicted: str,
    ) -> bool:

        return (
            not answer_correct
            and expected == "High confidence"
            and predicted == NO_ANSWER
        )

    # ------------------------------------------------------------------
    # Error classification
    # ------------------------------------------------------------------

    @staticmethod
    def _is_retryable_error(
        exc: Exception,
    ) -> bool:

        error_text = str(exc).lower()

        # Daily quota exhaustion must NOT be retried.
        daily_quota_terms = (
            "generate_requests_per_day",
            "perday",
            "daily quota exceeded",
            "daily_quota",
        )

        if any(
            term in error_text
            for term in daily_quota_terms
        ):
            return False

        # Transient provider errors and RPM rate limits can be retried.
        transient_terms = (
            "rate limit",
            "rate_limit_exceeded",
            "resource_exhausted",
            "quota exceeded",
            "free_tier_requests",
            "generate_content_free_tier_requests",
            "temporarily unavailable",
            "service unavailable",
            "503",
            "429",
            "internal server error",
            "deadline exceeded",
            "timeout",
            "timed out",
        )

        if any(
            term in error_text
            for term in transient_terms
        ):
            return True

        # Structured output parsing/validation can occasionally
        # fail transiently and can be retried.
        parsing_terms = (
            "json_validate_failed",
            "failed to generate json",
            "outputparserexception",
            "failed to parse",
            "max completion tokens reached",
        )

        if any(
            term in error_text
            for term in parsing_terms
        ):
            return True

        return False

    # ------------------------------------------------------------------
    # Benchmark execution
    # ------------------------------------------------------------------

    def run(
        self,
        questions: list[dict] | None = None,
    ) -> list[dict]:

        if questions is not None:
            self.questions = questions

        if not self.questions:
            raise ValueError(
                "No evaluation questions were provided."
            )

        results: list[dict] = []

        total = len(self.questions)

        for position, item in enumerate(
            self.questions,
            start=1,
        ):

            question_id = item["id"]

            print(
                f"[{position}/{total}] "
                f"{question_id}: {item['question']}"
            )

            try:

                result = None
                last_exception = None

                for attempt in range(
                    EVALUATION_RETRY_ATTEMPTS
                ):

                    try:

                        result = self.agent.run(
                            item["question"]
                        )

                        break

                    except Exception as exc:

                        last_exception = exc

                        is_retryable = (
                            self._is_retryable_error(
                                exc
                            )
                        )

                        if not is_retryable:
                            raise

                        if attempt == (
                            EVALUATION_RETRY_ATTEMPTS - 1
                        ):
                            raise

                        # Check if error specifies retry delay
                        exc_str = str(exc)
                        delay = EVALUATION_RETRY_BASE_DELAY * (attempt + 1)
                        
                        delay_match = re.search(r"retry in (\d+(?:\.\d+)?)s", exc_str, re.IGNORECASE)
                        if not delay_match:
                            delay_match = re.search(r"retryDelay':\s*'(\d+)s'", exc_str, re.IGNORECASE)
                        if delay_match:
                            delay = max(delay, int(float(delay_match.group(1))) + 2)

                        print(
                            f"  Transient provider error ({type(exc).__name__}). "
                            f"Retrying in {delay}s "
                            f"(attempt "
                            f"{attempt + 2}/"
                            f"{EVALUATION_RETRY_ATTEMPTS})..."
                        )

                        time.sleep(
                            delay
                        )

                if result is None:

                    if last_exception is not None:
                        raise last_exception

                    raise RuntimeError(
                        "Agent returned no result."
                    )

                # ------------------------------------------------------
                # Convert Pydantic verification result to JSON data.
                # ------------------------------------------------------

                verification = result.get(
                    "verification"
                )

                if hasattr(
                    verification,
                    "model_dump",
                ):
                    verification_data = (
                        verification.model_dump()
                    )

                elif isinstance(
                    verification,
                    dict,
                ):
                    verification_data = verification

                else:
                    verification_data = {}

                # ------------------------------------------------------
                # Independent benchmark correctness evaluation
                # ------------------------------------------------------

                answer_correct, note = (
                    self._answer_is_correct(
                        result,
                        item,
                    )
                )

                confidence_correct = (
                    self._confidence_is_correct(
                        result.get(
                            "confidence",
                            "",
                        ),
                        item["expected_confidence"],
                    )
                )

                dangerous = (
                    self._dangerous_overconfidence(
                        answer_correct,
                        result.get(
                            "confidence",
                            "",
                        ),
                    )
                )

                over_cautious = (
                    self._over_cautious_abstention(
                        answer_correct,
                        item["expected_confidence"],
                        result.get(
                            "confidence",
                            "",
                        ),
                    )
                )

                results.append(
                    {
                        "id": question_id,
                        "category": item["category"],
                        "question": item["question"],
                        "reference_answer": item[
                            "reference_answer"
                        ],
                        "expected_confidence": item[
                            "expected_confidence"
                        ],
                        "application_answer": result.get(
                            "answer",
                            "",
                        ),
                        "predicted_confidence": result.get(
                            "confidence",
                            "",
                        ),
                        "answer_correct": answer_correct,
                        "confidence_correct": confidence_correct,
                        "dangerous_overconfidence": dangerous,
                        "over_cautious_abstention": over_cautious,
                        "notes": note,
                        "reason": result.get(
                            "reason",
                            "",
                        ),
                        "evidence_ids": result.get(
                            "evidence_ids",
                            [],
                        ),
                        "verification": verification_data,
                    }
                )

                print(
                    f"  Answer correct: "
                    f"{answer_correct}"
                )

                print(
                    f"  Confidence: "
                    f"{result.get('confidence', '')}"
                )

                # ------------------------------------------------------
                # Pace benchmark requests.
                # ------------------------------------------------------

                if position < total:

                    print(
                        f"  Waiting "
                        f"{EVALUATION_QUESTION_DELAY}s "
                        f"before the next question..."
                    )

                    time.sleep(
                        EVALUATION_QUESTION_DELAY
                    )

            except Exception as exc:

                error_message = (
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"  ERROR: {error_message}"
                )

                # Execution failures are explicitly recorded but are
                # not treated as factual model decisions.
                results.append(
                    {
                        "id": question_id,
                        "category": item["category"],
                        "question": item["question"],
                        "reference_answer": item[
                            "reference_answer"
                        ],
                        "expected_confidence": item[
                            "expected_confidence"
                        ],
                        "application_answer": "",
                        "predicted_confidence": "",
                        "answer_correct": False,
                        "confidence_correct": False,
                        "dangerous_overconfidence": False,
                        "over_cautious_abstention": False,
                        "notes": (
                            "Execution error. This question was "
                            "excluded from model accuracy and "
                            "confidence calibration metrics."
                        ),
                        "error": error_message,
                        "reason": "",
                        "evidence_ids": [],
                        "verification": {},
                    }
                )

                # Don't immediately hammer the provider after an
                # execution failure.
                if position < total:
                    time.sleep(
                        EVALUATION_QUESTION_DELAY
                    )

        self.results = results

        # --------------------------------------------------------------
        # Calculate metrics after all questions have been processed.
        # --------------------------------------------------------------

        from app.evaluation.metrics import EvaluationMetrics

        self.metrics = EvaluationMetrics(
            self.results
        )

        return self.results