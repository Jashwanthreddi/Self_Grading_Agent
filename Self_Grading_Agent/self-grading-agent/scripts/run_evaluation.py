from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Application imports
# ---------------------------------------------------------------------------

from app.config import get_settings
from app.evaluation.calibration import CalibrationAnalyzer
from app.evaluation.evaluator import EvaluationRunner
from app.pipeline.agent import SelfGradingAgent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

settings = get_settings()

EVALUATION_PATH = Path(settings.evaluation_path)

OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"

RAW_RESULTS_PATH = OUTPUT_DIR / "evaluation_results.json"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
CSV_RESULTS_PATH = OUTPUT_DIR / "evaluation_results.csv"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confidence_confusion_matrix.png"
FAILURE_ANALYSIS_PATH = OUTPUT_DIR / "failure_analysis.json"


# ---------------------------------------------------------------------------
# Exact confidence labels
# ---------------------------------------------------------------------------

HIGH_LABEL = (
    "High confidence — the answer is clearly supported by the sources."
)

LOW_LABEL = (
    "Low confidence — the available evidence is incomplete, ambiguous, "
    "or requires an unsupported inference."
)

NO_ANSWER_LABEL = (
    "I don't know — the documents do not contain enough evidence to answer."
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def ensure_output_dir() -> None:
    """Create the results directory if it does not exist."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_questions(
    path: Path,
) -> list[dict[str, Any]]:
    """Load and validate the evaluation dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "questions" not in data:
            raise ValueError(
                "Evaluation JSON object must contain a 'questions' field."
            )

        data = data["questions"]

    if not isinstance(data, list):
        raise ValueError(
            "Evaluation dataset must contain a list of questions."
        )

    if not data:
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

    for item in data:

        if not isinstance(item, dict):
            raise ValueError(
                "Every evaluation question must be a JSON object."
            )

        missing = required_fields - set(item.keys())

        if missing:
            raise ValueError(
                f"Question {item.get('id', '<unknown>')} "
                f"is missing fields: {sorted(missing)}"
            )

    return data


def to_serializable(
    value: Any,
) -> Any:
    """
    Convert Pydantic models and nested application objects
    into JSON-compatible Python structures.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, list):
        return [
            to_serializable(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            to_serializable(item)
            for item in value
        ]

    if isinstance(value, dict):
        return {
            str(key): to_serializable(item)
            for key, item in value.items()
        }

    if hasattr(
        value,
        "model_dump",
    ):
        return to_serializable(
            value.model_dump()
        )

    if hasattr(
        value,
        "dict",
    ):
        return to_serializable(
            value.dict()
        )

    if hasattr(
        value,
        "__dict__",
    ):
        return to_serializable(
            vars(value)
        )

    return str(value)


def write_json(
    path: Path,
    data: Any,
) -> None:
    """Write JSON with readable formatting."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            to_serializable(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_csv(
    results: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Save one row per evaluation question.

    Execution failures are retained in the CSV so the benchmark
    remains fully auditable.
    """

    fieldnames = [
        "id",
        "category",
        "question",
        "reference_answer",
        "expected_confidence",
        "application_answer",
        "predicted_confidence",
        "answer_correct",
        "confidence_correct",
        "dangerous_overconfidence",
        "over_cautious_abstention",
        "notes",
        "reason",
        "evidence_ids",
        "error",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            evidence_ids = result.get(
                "evidence_ids",
                [],
            )

            if isinstance(
                evidence_ids,
                list,
            ):
                evidence_ids = "; ".join(
                    str(item)
                    for item in evidence_ids
                )

            writer.writerow(
                {
                    "id": result.get(
                        "id",
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
                    "reference_answer": result.get(
                        "reference_answer",
                        "",
                    ),
                    "expected_confidence": result.get(
                        "expected_confidence",
                        "",
                    ),
                    "application_answer": result.get(
                        "application_answer",
                        "",
                    ),
                    "predicted_confidence": result.get(
                        "predicted_confidence",
                        "",
                    ),
                    "answer_correct": result.get(
                        "answer_correct",
                        "",
                    ),
                    "confidence_correct": result.get(
                        "confidence_correct",
                        "",
                    ),
                    "dangerous_overconfidence": result.get(
                        "dangerous_overconfidence",
                        "",
                    ),
                    "over_cautious_abstention": result.get(
                        "over_cautious_abstention",
                        "",
                    ),
                    "notes": result.get(
                        "notes",
                        "",
                    ),
                    "reason": result.get(
                        "reason",
                        "",
                    ),
                    "evidence_ids": evidence_ids,
                    "error": result.get(
                        "error",
                        "",
                    ),
                }
            )


# ---------------------------------------------------------------------------
# Failure analysis
# ---------------------------------------------------------------------------

def build_failure_analysis(
    metrics: Any,
) -> dict[str, Any]:
    """
    Build the failure-analysis artifact required by the assignment.

    Failure categories:
        1. Dangerous overconfidence
        2. Incorrect answers
        3. Over-cautious abstentions

    Execution/provider failures are reported separately.
    """

    failures: list[dict[str, Any]] = []

    dangerous_cases = metrics.dangerous_cases()
    incorrect_answers = metrics.incorrect_answers()
    over_cautious_cases = metrics.over_cautious_cases()
    execution_failures = metrics.execution_failures()

    # ------------------------------------------------------------------
    # Dangerous overconfidence
    # ------------------------------------------------------------------

    for case in dangerous_cases:

        failures.append(
            {
                "failure_type": "dangerous_overconfidence",
                "question_id": case.get("id"),
                "question": case.get("question"),
                "response": case.get(
                    "application_answer",
                    "",
                ),
                "predicted_confidence": case.get(
                    "predicted_confidence",
                    "",
                ),
                "expected_category": case.get(
                    "category",
                    "",
                ),
                "reference_answer": case.get(
                    "reference_answer",
                    "",
                ),
                "evidence_ids": case.get(
                    "evidence_ids",
                    [],
                ),
                "verification": case.get(
                    "verification",
                    {},
                ),
                "what_went_wrong": (
                    "The application answer was judged incorrect "
                    "while the system assigned High confidence."
                ),
                "possible_cause": (
                    "Potential retrieval, generation, verification, "
                    "or confidence-threshold failure."
                ),
                "recommended_change": (
                    "Inspect the retrieved evidence and claim-level "
                    "verification. Tighten the confidence decision "
                    "if unsupported claims can reach High confidence."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Incorrect answers
    # ------------------------------------------------------------------

    for case in incorrect_answers:

        question_id = case.get("id")

        already_exists = any(
            failure.get("question_id") == question_id
            for failure in failures
        )

        if already_exists:
            continue

        failures.append(
            {
                "failure_type": "incorrect_answer",
                "question_id": question_id,
                "question": case.get("question"),
                "response": case.get(
                    "application_answer",
                    "",
                ),
                "predicted_confidence": case.get(
                    "predicted_confidence",
                    "",
                ),
                "expected_category": case.get(
                    "category",
                    "",
                ),
                "reference_answer": case.get(
                    "reference_answer",
                    "",
                ),
                "evidence_ids": case.get(
                    "evidence_ids",
                    [],
                ),
                "verification": case.get(
                    "verification",
                    {},
                ),
                "what_went_wrong": (
                    "The final application answer did not contain "
                    "the benchmark's required factual information."
                ),
                "possible_cause": (
                    "Potential retrieval, generation, verification, "
                    "or confidence-decision failure."
                ),
                "recommended_change": (
                    "Inspect evidence quality, generated claims, "
                    "and independent verification for this question."
                ),
            }
        )

    # ------------------------------------------------------------------
    # Over-cautious abstentions
    # ------------------------------------------------------------------

    for case in over_cautious_cases:

        question_id = case.get("id")

        already_exists = any(
            failure.get("question_id") == question_id
            for failure in failures
        )

        if already_exists:
            continue

        failures.append(
            {
                "failure_type": "over_cautious_abstention",
                "question_id": question_id,
                "question": case.get("question"),
                "response": case.get(
                    "application_answer",
                    "",
                ),
                "predicted_confidence": case.get(
                    "predicted_confidence",
                    "",
                ),
                "expected_category": case.get(
                    "category",
                    "",
                ),
                "reference_answer": case.get(
                    "reference_answer",
                    "",
                ),
                "evidence_ids": case.get(
                    "evidence_ids",
                    [],
                ),
                "verification": case.get(
                    "verification",
                    {},
                ),
                "what_went_wrong": (
                    "The system abstained even though the benchmark "
                    "contained sufficient evidence to answer."
                ),
                "possible_cause": (
                    "Potentially strict confidence threshold, "
                    "verification score, or evidence selection."
                ),
                "recommended_change": (
                    "Inspect whether the correct evidence was retrieved "
                    "and whether the verifier correctly recognized support."
                ),
            }
        )

    return {
        "failure_count": len(failures),
        "failures": failures,
        "execution_failures": execution_failures,
        "execution_failure_count": len(
            execution_failures
        ),
        "analysis_note": (
            "Execution/provider failures are kept separate from "
            "model failures and are excluded from accuracy and "
            "calibration denominators."
        ),
    }


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(
    metrics: Any,
) -> None:

    print()
    print("=" * 80)
    print("SELF-GRADING AGENT — EVALUATION SUMMARY")
    print("=" * 80)

    print(
        f"Total questions:          "
        f"{metrics.total_questions}"
    )

    print(
        f"Successful executions:    "
        f"{len(metrics._successful_results())}"
    )

    print(
        f"Execution failures:      "
        f"{len(metrics.execution_failures())}"
    )

    print(
        f"Answer accuracy:          "
        f"{metrics.answer_accuracy():.2f}%"
    )

    print(
        f"Confidence accuracy:     "
        f"{metrics.confidence_accuracy():.2f}%"
    )

    print()
    print("CONFIDENCE DISTRIBUTION")
    print("-" * 80)

    distribution = metrics.confidence_distribution()

    print(
        f"High confidence:         "
        f"{distribution.get(HIGH_LABEL, 0)}"
    )

    print(
        f"Low confidence:          "
        f"{distribution.get(LOW_LABEL, 0)}"
    )

    print(
        f"I don't know:             "
        f"{distribution.get(NO_ANSWER_LABEL, 0)}"
    )

    print()
    print("HIGH-CONFIDENCE SAFETY")
    print("-" * 80)

    print(
        f"High-confidence answers: "
        f"{metrics.high_confidence_count()}"
    )

    print(
        f"Correct high-confidence: "
        f"{metrics.correct_high_confidence_count()}"
    )

    print(
        f"High-confidence precision:"
        f" {metrics.high_confidence_precision():.2f}%"
    )

    print(
        f"Dangerous overconfidence:"
        f" {metrics.dangerous_overconfidence_count()}"
    )

    print()
    print("ABSTENTION")
    print("-" * 80)

    print(
        f"I don't know predictions: "
        f"{metrics.no_answer_count()}"
    )

    print(
        f"Over-cautious abstentions: "
        f"{metrics.over_cautious_abstention_count()}"
    )

    print()
    print("CONFIDENCE CONFUSION MATRIX")
    print("-" * 80)

    matrix = metrics.confidence_confusion_matrix()

    no_answer_short = "I don't know"

    print(
        f"{'Expected':<30}"
        f"{'High':>10}"
        f"{'Low':>10}"
        f"{no_answer_short:>16}"
    )

    print("-" * 66)

    label_pairs = [
        (
            HIGH_LABEL,
            "High",
        ),
        (
            LOW_LABEL,
            "Low",
        ),
        (
            NO_ANSWER_LABEL,
            no_answer_short,
        ),
    ]

    for expected_label, short_name in label_pairs:

        row = matrix.get(
            expected_label,
            {},
        )

        high_count = row.get(
            HIGH_LABEL,
            0,
        )

        low_count = row.get(
            LOW_LABEL,
            0,
        )

        no_answer_count = row.get(
            NO_ANSWER_LABEL,
            0,
        )

        print(
            f"{short_name:<30}"
            f"{high_count:>10}"
            f"{low_count:>10}"
            f"{no_answer_count:>16}"
        )

    print()
    print("CATEGORY METRICS")
    print("-" * 80)

    category_metrics = metrics.category_metrics()

    for category, values in category_metrics.items():

        print(
            f"{category}:"
        )

        print(
            f"  Questions:             "
            f"{values.get('questions', 0)}"
        )

        print(
            f"  Answer accuracy:       "
            f"{values.get('answer_accuracy', 0.0):.2f}%"
        )

        print(
            f"  Confidence accuracy:   "
            f"{values.get('confidence_accuracy', 0.0):.2f}%"
        )

        print(
            f"  Dangerous overconfidence:"
            f" {values.get('dangerous_overconfidence', 0)}"
        )

        print(
            f"  Over-cautious abstentions:"
            f" {values.get('over_cautious_abstentions', 0)}"
        )

    print()
    print("CALIBRATION")
    print("-" * 80)

    calibration = metrics.confidence_calibration()

    for level, values in calibration.items():

        print(
            f"{level}: "
            f"{values.get('predictions', 0)} predictions, "
            f"{values.get('correct', 0)} correct, "
            f"{values.get('correctness_percent', 0.0):.2f}% correct"
        )

    print()
    print("OUTPUT FILES")
    print("-" * 80)

    print(
        f"Raw results:             {RAW_RESULTS_PATH}"
    )

    print(
        f"Metrics:                 {METRICS_PATH}"
    )

    print(
        f"CSV results:             {CSV_RESULTS_PATH}"
    )

    print(
        f"Confusion matrix:        {CONFUSION_MATRIX_PATH}"
    )

    print(
        f"Failure analysis:        {FAILURE_ANALYSIS_PATH}"
    )

    print("=" * 80)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ensure_output_dir()

    print("=" * 80)
    print("SELF-GRADING AGENT — EVALUATION")
    print("=" * 80)

    print(
        f"Project root:             {PROJECT_ROOT}"
    )

    print(
        f"Evaluation dataset:      {EVALUATION_PATH}"
    )

    print(
        f"Output directory:        {OUTPUT_DIR}"
    )

    print()

    questions = load_questions(
        EVALUATION_PATH
    )

    print(
        f"Loaded evaluation questions: {len(questions)}"
    )

    print()
    print("Initializing Self-Grading Agent...")

    agent = SelfGradingAgent()

    print(
        "Agent initialized successfully."
    )

    print()
    print("Running evaluation...")
    print("-" * 80)

    # ------------------------------------------------------------------
    # Run benchmark
    # ------------------------------------------------------------------

    runner = EvaluationRunner(
        agent=agent,
        questions=questions,
    )

    results = runner.run()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    metrics = runner.metrics

    if metrics is None:
        raise RuntimeError(
            "Evaluation metrics were not generated."
        )

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------

    serializable_results = [
        to_serializable(result)
        for result in results
    ]

    write_json(
        RAW_RESULTS_PATH,
        serializable_results,
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------

    metrics_summary = metrics.summary()

    write_json(
        METRICS_PATH,
        metrics_summary,
    )

    # ------------------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------------------

    save_csv(
        serializable_results,
        CSV_RESULTS_PATH,
    )

    # ------------------------------------------------------------------
    # Save confidence confusion matrix
    # ------------------------------------------------------------------

    calibration = CalibrationAnalyzer(
        serializable_results
    )

    calibration.save_confusion_matrix(
        CONFUSION_MATRIX_PATH
    )

    # ------------------------------------------------------------------
    # Save failure analysis
    # ------------------------------------------------------------------

    failure_analysis = build_failure_analysis(
        metrics
    )

    write_json(
        FAILURE_ANALYSIS_PATH,
        failure_analysis,
    )

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------

    print_summary(
        metrics
    )

    print(
        "Evaluation completed successfully."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()