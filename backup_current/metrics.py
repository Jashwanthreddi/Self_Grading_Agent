from collections import Counter, defaultdict


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
    Calculate answer-quality and confidence-calibration metrics.

    Execution failures are excluded from quality denominators.

    Result schema expected from EvaluationRunner:

        question_id
        category
        question
        reference_answer
        expected_confidence
        predicted_answer
        predicted_confidence
        correct
        confidence_correct
        dangerous_overconfidence
        over_cautious_abstention
        correctness_notes
        retrieved_evidence
        verification
        execution_error
    """

    def __init__(self, results: list[dict]):
        if not isinstance(results, list):
            raise TypeError(
                "Evaluation results must be a list."
            )

        self.results = results

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------

    @property
    def total_questions(self) -> int:
        return len(self.results)

    def _successful_results(self) -> list[dict]:
        """
        Return only questions that completed successfully.

        Execution failures are reported separately and are never treated
        as incorrect model answers.
        """

        return [
            result
            for result in self.results
            if not result.get("execution_error")
        ]

    @staticmethod
    def _percentage(
        numerator: int,
        denominator: int,
    ) -> float:

        if denominator == 0:
            return 0.0

        return round(
            (numerator / denominator) * 100,
            2,
        )

    # ------------------------------------------------------------------
    # Answer metrics
    # ------------------------------------------------------------------

    def answer_accuracy(self) -> float:
        """
        Percentage of successfully evaluated questions with a
        factually correct answer.
        """

        results = self._successful_results()

        if not results:
            return 0.0

        correct = sum(
            bool(result.get("correct", False))
            for result in results
        )

        return self._percentage(
            correct,
            len(results),
        )

    # ------------------------------------------------------------------
    # Confidence metrics
    # ------------------------------------------------------------------

    def confidence_accuracy(self) -> float:
        """
        Percentage of successful questions where the predicted
        confidence matches the benchmark's expected confidence.
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

    # ------------------------------------------------------------------
    # High-confidence calibration
    # ------------------------------------------------------------------

    def high_confidence_count(self) -> int:

        return sum(
            result.get("predicted_confidence")
            == HIGH_CONFIDENCE
            for result in self._successful_results()
        )

    def correct_high_confidence_count(self) -> int:

        return sum(
            result.get("predicted_confidence")
            == HIGH_CONFIDENCE
            and result.get("correct", False)
            for result in self._successful_results()
        )

    def high_confidence_precision(self) -> float:
        """
        Among High-confidence predictions, how many were actually
        factually correct?

        This is the key dangerous-overconfidence metric.
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
        Count incorrect answers that were given High confidence.
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

    # ------------------------------------------------------------------
    # Abstention metrics
    # ------------------------------------------------------------------

    def no_answer_count(self) -> int:

        return sum(
            result.get("predicted_confidence")
            == NO_ANSWER
            for result in self._successful_results()
        )

    def over_cautious_abstention_count(self) -> int:
        """
        Count cases where the system said I don't know even though
        the benchmark answer was correct.
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

    # ------------------------------------------------------------------
    # Confidence distribution
    # ------------------------------------------------------------------

    def confidence_distribution(self) -> dict[str, int]:

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

    # ------------------------------------------------------------------
    # Confidence confusion matrix
    # ------------------------------------------------------------------

    def confidence_confusion_matrix(self) -> dict:
        """
        Build expected-vs-predicted confidence matrix.

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
                "expected_confidence",
                "",
            )

            expected = expected_mapping.get(
                expected_name,
                expected_name,
            )

            predicted = result.get(
                "predicted_confidence",
                "",
            )

            if (
                expected in matrix
                and predicted in matrix[expected]
            ):
                matrix[expected][predicted] += 1

        return matrix

    # ------------------------------------------------------------------
    # Per-category metrics
    # ------------------------------------------------------------------

    def category_metrics(self) -> dict:
        """
        Calculate metrics separately for answerable, unanswerable,
        partially-supported/ambiguous, and trap questions.
        """

        grouped = defaultdict(list)

        for result in self._successful_results():

            grouped[
                result.get(
                    "category",
                    "unknown",
                )
            ].append(result)

        output = {}

        for category, results in grouped.items():

            total = len(results)

            answer_correct = sum(
                bool(
                    result.get(
                        "correct",
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
                "questions": total,
                "answer_accuracy_percent": self._percentage(
                    answer_correct,
                    total,
                ),
                "confidence_accuracy_percent": self._percentage(
                    confidence_correct,
                    total,
                ),
                "dangerous_overconfidence": dangerous,
                "over_cautious_abstentions": abstentions,
            }

        return dict(output)

    # ------------------------------------------------------------------
    # Execution failures
    # ------------------------------------------------------------------

    def execution_failures(self) -> list[dict]:
        """
        Return infrastructure/API failures separately.
        """

        return [
            {
                "question_id": result.get(
                    "question_id",
                    "",
                ),
                "category": result.get(
                    "category",
                    "",
                ),
                "question": result.get(
                    "question",
                    "",
                ),
                "error": result.get(
                    "execution_error",
                    "",
                ),
            }
            for result in self.results
            if result.get("execution_error")
        ]

    # ------------------------------------------------------------------
    # Incorrect answers
    # ------------------------------------------------------------------

    def incorrect_answers(self) -> list[dict]:

        return [
            result
            for result in self._successful_results()
            if not result.get(
                "correct",
                False,
            )
        ]

    # ------------------------------------------------------------------
    # Dangerous cases
    # ------------------------------------------------------------------

    def dangerous_cases(self) -> list[dict]:

        return [
            result
            for result in self._successful_results()
            if result.get(
                "dangerous_overconfidence",
                False,
            )
        ]

    # ------------------------------------------------------------------
    # Over-cautious cases
    # ------------------------------------------------------------------

    def over_cautious_cases(self) -> list[dict]:

        return [
            result
            for result in self._successful_results()
            if result.get(
                "over_cautious_abstention",
                False,
            )
        ]

    # ------------------------------------------------------------------
    # Overall summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Produce the complete evaluation report.

        Important:
        quality metrics use only successfully executed questions.
        Execution failures are reported separately.
        """

        successful = self._successful_results()
        failures = self.execution_failures()

        return {
            "total_questions": self.total_questions,

            "successful_questions": len(
                successful
            ),

            "execution_failures": len(
                failures
            ),

            "answer_accuracy_percent": (
                self.answer_accuracy()
            ),

            "confidence_accuracy_percent": (
                self.confidence_accuracy()
            ),

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

            "i_dont_know_count": (
                self.no_answer_count()
            ),

            "over_cautious_abstentions": (
                self.over_cautious_abstention_count()
            ),

            "confidence_distribution": (
                self.confidence_distribution()
            ),

            "confidence_confusion_matrix": (
                self.confidence_confusion_matrix()
            ),

            "category_metrics": (
                self.category_metrics()
            ),

            "execution_failure_details": (
                failures
            ),
        }
