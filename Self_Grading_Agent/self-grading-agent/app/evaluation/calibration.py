from pathlib import Path

import matplotlib.pyplot as plt

from app.evaluation.metrics import EvaluationMetrics


class CalibrationAnalyzer:
    """
    Generates calibration-focused evaluation artifacts.

    The confusion matrix compares the expected confidence category
    from the benchmark with the confidence predicted by the agent.
    """

    def __init__(self, results: list[dict]):
        self.results = results
        self.metrics = EvaluationMetrics(results)

    def confusion_matrix(self) -> dict:
        return self.metrics.confidence_confusion_matrix()

    def save_confusion_matrix(self, output_path: Path) -> None:
        data = self.confusion_matrix()

        labels = [
            "High confidence",
            "Low confidence",
            "I don't know",
        ]

        confidence_values = [
            "High confidence — the answer is clearly supported by the sources.",
            (
                "Low confidence — the available evidence is incomplete, "
                "ambiguous, or requires an unsupported inference."
            ),
            (
                "I don't know — the documents do not contain enough "
                "evidence to answer."
            ),
        ]

        matrix = []

        for expected in confidence_values:
            row = []

            for predicted in confidence_values:
                row.append(
                    data.get(expected, {}).get(predicted, 0)
                )

            matrix.append(row)

        figure, axis = plt.subplots(figsize=(8, 6))

        image = axis.imshow(matrix)

        axis.set_xticks(range(len(labels)))
        axis.set_yticks(range(len(labels)))

        axis.set_xticklabels(labels)
        axis.set_yticklabels(labels)

        axis.set_xlabel("Predicted confidence")
        axis.set_ylabel("Expected confidence")
        axis.set_title("Confidence Calibration Confusion Matrix")

        for row_index in range(len(matrix)):
            for column_index in range(len(matrix[row_index])):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index][column_index]),
                    ha="center",
                    va="center",
                )

        figure.tight_layout()

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            output_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(figure)


# Backward compatibility for any existing code that imports
# CalibrationEvaluator.
CalibrationEvaluator = CalibrationAnalyzer