from __future__ import annotations

from collections import Counter, defaultdict


# ---------------------------------------------------------------------
# EXACT CONFIDENCE LABELS
# ---------------------------------------------------------------------

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


CONFIDENCE_LABELS = [
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    NO_ANSWER,
]


CONFIDENCE_SHORT_NAMES = {
    HIGH_CONFIDENCE: "High",
    LOW_CONFIDENCE: "Low",
    NO_ANSWER: "I don't know",
}


class EvaluationMetrics:
    """
    Calculate evaluation and calibration metrics for the
    self-grading QA agent.

    The evaluator distinguishes between:

    1. Answer correctness
    2. Confidence correctness
    3. High-confidence precision
    4. Dangerous overconfidence
    5. Abstention behavior
    6. Confidence distribution
    7. Confidence confusion matrix
    8. Category-level performance
    9. Execution failures

    Infrastructure/execution failures are excluded from accuracy
    and calibration denominators but are reported separately.
    """

    def __init__(self, results: list[dict]):
        if not isinstance(results, list):
            raise TypeError(
                "Evaluation results must be a list."
            )

        self.results = results

    # -----------------------------------------------------------------
    # BASIC HELPERS
    # -----------------------------------------------------------------

    @property
    def total_questions(self) -> int:
        return len(self.results)

    def _successful_results(self) -> list[dict]:
        """
        Return only questions that completed successfully.

        Execution failures are excluded from accuracy/calibration
        denominators because there is no model prediction to evaluate.
        """

        return [
            result
            for result in self.results
            if not result.get("error")
        ]

    def _count_true(
        self,
        field: str,
    ) -> int:
        """
        Count successful results where a boolean field is True.
        """

        return sum(
            bool(result.get(field, False))
            for result in self._successful_results()
        )

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
    ) -> float:
        """
        Return percentage rounded to two decimal places.
        """

        if denominator == 0:
            return 0.0

        return round(
            (numerator / denominator) * 100,
            2,
        )

    # -----------------------------------------------------------------
    # ANSWER METRICS
    # -----------------------------------------------------------------

    def answer_accuracy(self) -> float:
        """
        Percentage of successfully processed questions whose
        application answer was judged correct.
        """

        results = self._successful_results()

        if not results:
            return 0.0

        correct = sum(
            bool(result.get("answer_correct", False))
            for result in results
        )

        return self._percentage(
            correct,
            len(results),
        )

    def confidence_accuracy(self) -> float:
        """
        Percentage of successfully processed questions where the
        predicted confidence label matched the expected confidence.
        """

        results = self._successful_results()

        if not results:
            return 0.0

        correct = sum(
            bool(result.get("confidence_correct", False))
            for result in results
        )

        return self._percentage(
            correct,
            len(results),
        )

    # -----------------------------------------------------------------
    # HIGH-CONFIDENCE CALIBRATION
    # -----------------------------------------------------------------

    def high_confidence_count(self) -> int:
        """
        Number of successful questions assigned High confidence.
        """

        return sum(
            result.get("predicted_confidence")
            == HIGH_CONFIDENCE
            for result in self._successful_results()
        )

    def correct_high_confidence_count(self) -> int:
        """
        Number of High-confidence predictions whose answers were
        actually judged correct.
        """

        return sum(
            result.get("predicted_confidence")
            == HIGH_CONFIDENCE
            and bool(result.get("answer_correct", False))
            for result in self._successful_results()
        )

    def high_confidence_precision(self) -> float:
        """
        Among High-confidence predictions, percentage that were correct.

        This is particularly useful for detecting dangerous
        overconfidence.
        """

        total_high = self.high_confidence_count()

        if total_high == 0:
            return 0.0

        correct_high = (
            self.correct_high_confidence_count()
        )

        return self._percentage(
            correct_high,
            total_high,
        )

    def dangerous_overconfidence_count(self) -> int:
        """
        Count incorrect answers that were assigned High confidence.

        This is one of the most important reliability metrics.
        """

        return sum(
            bool(
                result.get(
                    "dangerous_overconfidence",
                    False,
                )
            )
            for result in self._successful_results()
        )

    # -----------------------------------------------------------------
    # ABSTENTION METRICS
    # -----------------------------------------------------------------

    def no_answer_count(self) -> int:
        """
        Number of successful questions where the agent abstained
        with the exact I-don't-know confidence label.
        """

        return sum(
            result.get("predicted_confidence")
            == NO_ANSWER
            for result in self._successful_results()
        )

    def over_cautious_abstention_count(self) -> int:
        """
        Count cases where the system abstained even though the
        documents contained enough evidence to answer.
        """

        return sum(
            bool(
                result.get(
                    "over_cautious_abstention",
                    False,
                )
            )
            for result in self._successful_results()
        )

    # -----------------------------------------------------------------
    # CONFIDENCE DISTRIBUTION
    # -----------------------------------------------------------------

    def confidence_distribution(self) -> dict[str, int]:
        """
        Distribution of predicted confidence labels.
        """

        counts = Counter(
            result.get(
                "predicted_confidence",
                "",
            )
            for result in self._successful_results()
        )

        return {
            "High confidence": counts.get(
                HIGH_CONFIDENCE,
                0,
            ),
            "Low confidence": counts.get(
                LOW_CONFIDENCE,
                0,
            ),
            "I don't know": counts.get(
                NO_ANSWER,
                0,
            ),
        }

    # -----------------------------------------------------------------
    # CONFIDENCE CONFUSION MATRIX
    # -----------------------------------------------------------------

    def confidence_confusion_matrix(self) -> dict:
        """
        Build an expected-vs-predicted confidence matrix.

        Rows:
            expected confidence

        Columns:
            predicted confidence
        """

        matrix = {
            expected: {
                predicted: 0
                for predicted in CONFIDENCE_LABELS
            }
            for expected in CONFIDENCE_LABELS
        }

        expected_mapping = {
            "High confidence": HIGH_CONFIDENCE,
            "Low confidence": LOW_CONFIDENCE,
            "I don't know": NO_ANSWER,
        }

        for result in self._successful_results():

            expected_name = result.get(
                "expected_confidence"
            )

            predicted = result.get(
                "predicted_confidence"
            )

            expected = expected_mapping.get(
                expected_name
            )

            if (
                expected in matrix
                and predicted in matrix[expected]
            ):
                matrix[expected][predicted] += 1

        return matrix

    # -----------------------------------------------------------------
    # CATEGORY METRICS
    # -----------------------------------------------------------------

    def category_metrics(self) -> dict:
        """
        Calculate answer/confidence metrics independently for each
        evaluation category.

        Expected categories include:

            answerable
            unanswerable
            partial
            trap
        """

        grouped = defaultdict(list)

        for result in self._successful_results():

            category = result.get(
                "category",
                "unknown",
            )

            grouped[category].append(
                result
            )

        output = {}

        for category, results in grouped.items():

            answer_correct = sum(
                bool(
                    result.get(
                        "answer_correct",
                        False,
                    )
                )
                for result in results
            )

            confidence_correct = sum(
                bool(
                    result.get(
                        "confidence_correct",
                        False,
                    )
                )
                for result in results
            )

            dangerous = sum(
                bool(
                    result.get(
                        "dangerous_overconfidence",
                        False,
                    )
                )
                for result in results
            )

            abstentions = sum(
                bool(
                    result.get(
                        "over_cautious_abstention",
                        False,
                    )
                )
                for result in results
            )

            output[category] = {
                "questions": len(results),

                "answer_accuracy": self._percentage(
                    answer_correct,
                    len(results),
                ),

                "confidence_accuracy": self._percentage(
                    confidence_correct,
                    len(results),
                ),

                "dangerous_overconfidence": dangerous,

                "over_cautious_abstentions": abstentions,
            }

        return dict(output)

    # -----------------------------------------------------------------
    # ERROR ANALYSIS
    # -----------------------------------------------------------------

    def execution_failures(self) -> list[dict]:
        """
        Return details for questions that failed during execution.
        """

        return [
            {
                "id": result.get("id"),
                "question": result.get("question"),
                "error": result.get("error"),
            }
            for result in self.results
            if result.get("error")
        ]

    def incorrect_answers(self) -> list[dict]:
        """
        Return all successfully executed questions whose answer
        was judged incorrect.
        """

        return [
            result
            for result in self._successful_results()
            if not result.get(
                "answer_correct",
                False,
            )
        ]

    def dangerous_cases(self) -> list[dict]:
        """
        Return all dangerous overconfidence cases.
        """

        return [
            result
            for result in self._successful_results()
            if result.get(
                "dangerous_overconfidence",
                False,
            )
        ]

    def over_cautious_cases(self) -> list[dict]:
        """
        Return all over-cautious abstention cases.
        """

        return [
            result
            for result in self._successful_results()
            if result.get(
                "over_cautious_abstention",
                False,
            )
        ]

    # -----------------------------------------------------------------
    # CALIBRATION ERROR BREAKDOWN
    # -----------------------------------------------------------------

    def confidence_calibration(self) -> dict:
        """
        Provide simple calibration statistics for each predicted
        confidence level.

        For each confidence level:

            predictions
            correct
            correctness_percent

        This complements the confusion matrix.
        """

        calibration = {}

        successful = self._successful_results()

        for label in CONFIDENCE_LABELS:

            predictions = [
                result
                for result in successful
                if result.get(
                    "predicted_confidence"
                ) == label
            ]

            correct = sum(
                bool(
                    result.get(
                        "answer_correct",
                        False,
                    )
                )
                for result in predictions
            )

            calibration[
                CONFIDENCE_SHORT_NAMES[label]
            ] = {
                "predictions": len(predictions),
                "correct": correct,
                "correctness_percent": self._percentage(
                    correct,
                    len(predictions),
                ),
            }

        return calibration

    # -----------------------------------------------------------------
    # OVERALL SUMMARY
    # -----------------------------------------------------------------

    def summary(self) -> dict:
        """
        Produce the complete evaluation summary.

        This dictionary is designed to be serializable directly
        into metrics.json.
        """

        successful = self._successful_results()
        failures = self.execution_failures()

        return {
            # ---------------------------------------------------------
            # Dataset / execution
            # ---------------------------------------------------------

            "total_questions": self.total_questions,

            "successful_questions": len(
                successful
            ),

            "execution_failures": len(
                failures
            ),

            # ---------------------------------------------------------
            # Main metrics
            # ---------------------------------------------------------

            "answer_accuracy_percent": (
                self.answer_accuracy()
            ),

            "confidence_accuracy_percent": (
                self.confidence_accuracy()
            ),

            # ---------------------------------------------------------
            # High-confidence calibration
            # ---------------------------------------------------------

            "high_confidence_count": (
                self.high_confidence_count()
            ),

            "correct_high_confidence_count": (
                self.correct_high_confidence_count()
            ),

            "high_confidence_precision_percent": (
                self.high_confidence_precision()
            ),

            "dangerous_overconfidence": (
                self.dangerous_overconfidence_count()
            ),

            # ---------------------------------------------------------
            # Abstention
            # ---------------------------------------------------------

            "i_dont_know_count": (
                self.no_answer_count()
            ),

            "over_cautious_abstentions": (
                self.over_cautious_abstention_count()
            ),

            # ---------------------------------------------------------
            # Confidence distribution
            # ---------------------------------------------------------

            "confidence_distribution": (
                self.confidence_distribution()
            ),

            # ---------------------------------------------------------
            # Calibration
            # ---------------------------------------------------------

            "confidence_calibration": (
                self.confidence_calibration()
            ),

            "confidence_confusion_matrix": (
                self.confidence_confusion_matrix()
            ),

            # ---------------------------------------------------------
            # Category analysis
            # ---------------------------------------------------------

            "category_metrics": (
                self.category_metrics()
            ),

            # ---------------------------------------------------------
            # Failure analysis support
            # ---------------------------------------------------------

            "execution_failure_details": failures,

            "incorrect_answer_count": len(
                self.incorrect_answers()
            ),

            "dangerous_case_count": len(
                self.dangerous_cases()
            ),

            "over_cautious_case_count": len(
                self.over_cautious_cases()
            ),
        }