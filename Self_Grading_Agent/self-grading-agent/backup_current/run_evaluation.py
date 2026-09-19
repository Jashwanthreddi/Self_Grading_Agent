import csv
import json
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Project import path
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Application imports
# ----------------------------------------------------------------------

from app.config import get_settings
from app.decision.confidence import ConfidenceDecision
from app.evaluation.calibration import CalibrationAnalyzer
from app.evaluation.evaluator import (
    EvaluationRunner,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    NO_ANSWER,
)
from app.evaluation.metrics import EvaluationMetrics
from app.generation.generator import AnswerGenerator
from app.pipeline.agent import SelfGradingAgent
from app.retrieval.retriever import HybridRetriever
from app.verification.verifier import AnswerVerifier


# ----------------------------------------------------------------------
# Output directory
# ----------------------------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "evaluation_results"


# ----------------------------------------------------------------------
# Agent construction
# ----------------------------------------------------------------------

def build_agent() -> SelfGradingAgent:
    """Build the production self-grading pipeline."""

    settings = get_settings()

    retriever = HybridRetriever(
        faiss_index_path=settings.faiss_index_path,
        bm25_index_path=settings.bm25_index_path,
        embedding_model=settings.embedding_model,
        dense_top_k=settings.dense_top_k,
        bm25_top_k=settings.bm25_top_k,
        final_top_k=settings.final_top_k,
        dense_weight=settings.dense_weight,
        bm25_weight=settings.bm25_weight,
    )

    generator = AnswerGenerator()
    verifier = AnswerVerifier()
    decision = ConfidenceDecision()

    return SelfGradingAgent(
        retriever=retriever,
        generator=generator,
        verifier=verifier,
        decision=decision,
    )


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------

def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


# ----------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------

def save_csv(
    path: Path,
    results: list[dict],
) -> None:
    """
    Save a flattened benchmark report.

    The column names intentionally match EvaluationRunner's result
    schema so JSON and CSV remain consistent.
    """

    if not results:
        return

    rows = []

    for result in results:

        verification = result.get(
            "verification",
            {},
        ) or {}

        retrieved_evidence = result.get(
            "retrieved_evidence",
            [],
        ) or []

        rows.append(
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
                "reference_answer": result.get(
                    "reference_answer",
                    "",
                ),
                "expected_confidence": result.get(
                    "expected_confidence",
                    "",
                ),
                "predicted_answer": result.get(
                    "predicted_answer",
                    "",
                ),
                "predicted_confidence": result.get(
                    "predicted_confidence",
                    "",
                ),
                "correct": result.get(
                    "correct",
                    False,
                ),
                "confidence_correct": result.get(
                    "confidence_correct",
                    False,
                ),
                "dangerous_overconfidence": result.get(
                    "dangerous_overconfidence",
                    False,
                ),
                "over_cautious_abstention": result.get(
                    "over_cautious_abstention",
                    False,
                ),
                "correctness_notes": result.get(
                    "correctness_notes",
                    "",
                ),
                "retrieved_evidence": json.dumps(
                    retrieved_evidence,
                    ensure_ascii=False,
                ),
                "supported": verification.get(
                    "supported",
                    "",
                ),
                "support_score": verification.get(
                    "support_score",
                    "",
                ),
                "entity_match": verification.get(
                    "entity_match",
                    "",
                ),
                "temporal_match": verification.get(
                    "temporal_match",
                    "",
                ),
                "numerical_match": verification.get(
                    "numerical_match",
                    "",
                ),
                "condition_match": verification.get(
                    "condition_match",
                    "",
                ),
                "contradiction": verification.get(
                    "contradiction",
                    "",
                ),
                "evidence_sufficient": verification.get(
                    "evidence_sufficient",
                    "",
                ),
                "unsupported_inference": verification.get(
                    "unsupported_inference",
                    "",
                ),
                "verification_reason": verification.get(
                    "reason",
                    "",
                ),
                "execution_error": result.get(
                    "execution_error",
                    "",
                ),
            }
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------

def print_summary(summary: dict) -> None:
    """Print a human-readable evaluation summary."""

    print()
    print("=" * 70)
    print("SELF-GRADING AGENT — EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"Total questions: "
        f"{summary.get('total_questions', 0)}"
    )

    print(
        f"Successful questions: "
        f"{summary.get('successful_questions', 0)}"
    )

    print(
        f"Execution failures: "
        f"{summary.get('execution_failures', 0)}"
    )

    print()

    print(
        f"Answer accuracy: "
        f"{summary.get('answer_accuracy_percent', 0):.2f}%"
    )

    print(
        f"Confidence accuracy: "
        f"{summary.get('confidence_accuracy_percent', 0):.2f}%"
    )

    print()

    print(
        f"High-confidence predictions: "
        f"{summary.get('high_confidence_count', 0)}"
    )

    print(
        f"Correct high-confidence predictions: "
        f"{summary.get('correct_high_confidence_count', 0)}"
    )

    print(
        f"High-confidence precision: "
        f"{summary.get('high_confidence_precision_percent', 0):.2f}%"
    )

    print(
        f"Dangerous overconfidence: "
        f"{summary.get('dangerous_overconfidence', 0)}"
    )

    print()

    print(
        f"I don't know predictions: "
        f"{summary.get('i_dont_know_count', 0)}"
    )

    print(
        f"Over-cautious abstentions: "
        f"{summary.get('over_cautious_abstentions', 0)}"
    )

    print()

    print("Confidence distribution:")

    distribution = summary.get(
        "confidence_distribution",
        {},
    )

    for label, count in distribution.items():
        print(f"  {label}: {count}")

    print()

    print("Confidence confusion matrix:")

    matrix = summary.get(
        "confidence_confusion_matrix",
        {},
    )

    short_names = {
        HIGH_CONFIDENCE: "High",
        LOW_CONFIDENCE: "Low",
        NO_ANSWER: "I don't know",
    }

    print(
        f"{'Expected':<25}"
        f"{'High':>10}"
        f"{'Low':>10}"
        f"{'I don’t know':>15}"
    )

    for expected, row in matrix.items():

        print(
            f"{short_names.get(expected, expected):<25}"
            f"{row.get(HIGH_CONFIDENCE, 0):>10}"
            f"{row.get(LOW_CONFIDENCE, 0):>10}"
            f"{row.get(NO_ANSWER, 0):>15}"
        )

    print()

    print("Category metrics:")

    category_metrics = summary.get(
        "category_metrics",
        {},
    )

    for category, metrics in category_metrics.items():

        print()
        print(f"  {category}")

        print(
            f"    Questions: "
            f"{metrics.get('questions', 0)}"
        )

        print(
            f"    Answer accuracy: "
            f"{metrics.get('answer_accuracy_percent', 0):.2f}%"
        )

        print(
            f"    Confidence accuracy: "
            f"{metrics.get('confidence_accuracy_percent', 0):.2f}%"
        )

        print(
            f"    Dangerous overconfidence: "
            f"{metrics.get('dangerous_overconfidence', 0)}"
        )

        print(
            f"    Over-cautious abstentions: "
            f"{metrics.get('over_cautious_abstentions', 0)}"
        )

    print()
    print("=" * 70)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("SELF-GRADING AGENT — BENCHMARK EVALUATION")
    print("=" * 70)

    settings = get_settings()

    dataset_path = (
        PROJECT_ROOT
        / settings.evaluation_path
    )

    print(
        f"Dataset: {dataset_path}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    print()
    print("Initializing agent...")

    agent = build_agent()

    print("Agent initialized.")
    print()

    runner = EvaluationRunner(agent)

    questions = runner.load_questions(
        dataset_path
    )

    print(
        f"Loaded {len(questions)} benchmark questions."
    )

    print()

    results = runner.run(questions)

    # --------------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------------

    raw_results_path = (
        OUTPUT_DIR
        / "evaluation_results.json"
    )

    save_json(
        raw_results_path,
        results,
    )

    # --------------------------------------------------------------
    # Metrics
    # --------------------------------------------------------------

    metrics = EvaluationMetrics(results)

    summary = metrics.summary()

    metrics_path = (
        OUTPUT_DIR
        / "metrics.json"
    )

    save_json(
        metrics_path,
        summary,
    )

    # --------------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------------

    calibration = CalibrationAnalyzer(results)

    confusion_path = (
        OUTPUT_DIR
        / "confidence_confusion_matrix.png"
    )

    calibration.save_confusion_matrix(
        confusion_path
    )

    # --------------------------------------------------------------
    # CSV
    # --------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "evaluation_results.csv"
    )

    save_csv(
        csv_path,
        results,
    )

    # --------------------------------------------------------------
    # Console output
    # --------------------------------------------------------------

    print_summary(summary)

    print()
    print(
        f"Raw results: {raw_results_path}"
    )

    print(
        f"CSV results: {csv_path}"
    )

    print(
        f"Metrics: {metrics_path}"
    )

    print(
        f"Confusion matrix: {confusion_path}"
    )


if __name__ == "__main__":
    main()
