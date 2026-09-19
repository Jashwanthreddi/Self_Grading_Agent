import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json

import streamlit as st

from app.main import build_agent


st.set_page_config(
    page_title="Self-Grading QA Agent",
    page_icon="🤖",
    layout="wide",
)


@st.cache_resource
def get_agent():
    return build_agent()


def show_result(result: dict) -> None:
    st.subheader("Answer")

    st.write(result["answer"])

    confidence = result["confidence"]

    if confidence.startswith(
        "High confidence"
    ):
        st.success(confidence)

    elif confidence.startswith(
        "Low confidence"
    ):
        st.warning(confidence)

    else:
        st.error(confidence)

    st.caption(
        f"Reason: {result.get('reason', '')}"
    )

    st.subheader("Evidence")

    passages = result.get(
        "passages",
        [],
    )

    if not passages:
        st.info(
            "No evidence passages were retrieved."
        )
    else:
        for index, passage in enumerate(
            passages,
            start=1,
        ):
            with st.expander(
                f"{index}. "
                f"{passage.document_name} "
                f"({passage.retrieval_method}, "
                f"score={passage.score:.4f})"
            ):
                st.caption(
                    f"Document ID: "
                    f"{passage.document_id}"
                )

                st.write(
                    passage.content
                )

    verification = result.get(
        "verification"
    )

    if verification:
        with st.expander(
            "Independent Verification Details"
        ):
            st.write(
                {
                    "supported": verification.supported,
                    "support_score": (
                        verification.support_score
                    ),
                    "entity_match": (
                        verification.entity_match
                    ),
                    "temporal_match": (
                        verification.temporal_match
                    ),
                    "numerical_match": (
                        verification.numerical_match
                    ),
                    "condition_match": (
                        verification.condition_match
                    ),
                    "contradiction": (
                        verification.contradiction
                    ),
                    "evidence_sufficient": (
                        verification.evidence_sufficient
                    ),
                    "unsupported_inference": (
                        verification.unsupported_inference
                    ),
                }
            )


def show_evaluation() -> None:
    st.subheader(
        "Benchmark Evaluation Results"
    )

    results_path = Path(
        "evaluation_results/"
        "evaluation_results.json"
    )

    metrics_path = Path(
        "evaluation_results/"
        "metrics.json"
    )

    matrix_path = Path(
        "evaluation_results/"
        "confidence_confusion_matrix.png"
    )

    if not results_path.exists():
        st.info(
            "Run the evaluation harness first."
        )
        return

    with results_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    if metrics_path.exists():

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        columns = st.columns(5)

        columns[0].metric(
            "Questions",
            metrics.get(
                "total_questions",
                0,
            ),
        )

        columns[1].metric(
            "Successful",
            metrics.get(
                "successful_questions",
                0,
            ),
        )

        columns[2].metric(
            "Answer Accuracy",
            (
                f"{metrics.get('answer_accuracy_percent', 0):.1f}%"
            ),
        )

        columns[3].metric(
            "Over-cautious",
            metrics.get(
                "over_cautious_abstentions",
                0,
            ),
        )

        columns[4].metric(
            "Dangerous Overconfidence",
            metrics.get(
                "dangerous_overconfidence",
                0,
            ),
        )

        if matrix_path.exists():
            st.image(
                str(matrix_path),
                caption=(
                    "Confidence Calibration "
                    "Confusion Matrix"
                ),
            )

    table = []

    for item in results:
        table.append(
            {
                "ID": item.get("id", ""),
                "Category": item.get(
                    "category",
                    "",
                ),
                "Question": item.get(
                    "question",
                    "",
                ),
                "Expected": item.get(
                    "expected_confidence",
                    "",
                ),
                "Predicted": item.get(
                    "predicted_confidence",
                    "",
                ),
                "Correct": item.get(
                    "answer_correct",
                    False,
                ),
            }
        )

    st.dataframe(
        table,
        use_container_width=True,
    )


def main() -> None:

    st.title(
        "🤖 Self-Grading QA Agent"
    )

    st.write(
        "Evidence-grounded question answering "
        "with independent verification and "
        "calibrated confidence."
    )

    tab_qa, tab_eval = st.tabs(
        [
            "Ask a Question",
            "Evaluation",
        ]
    )

    with tab_qa:

        question = st.text_input(
            "Enter your question",
            placeholder=(
                "When was NovaSearch launched?"
            ),
        )

        if st.button(
            "Run Agent",
            type="primary",
        ):

            if not question.strip():

                st.warning(
                    "Please enter a question."
                )

            else:

                with st.spinner(
                    "Retrieving, generating, "
                    "and independently verifying..."
                ):

                    try:

                        result = (
                            get_agent().run(
                                question
                            )
                        )

                        show_result(
                            result
                        )

                    except Exception as exc:

                        st.error(
                            "The agent could not "
                            "complete the request."
                        )

                        st.exception(exc)

    with tab_eval:
        show_evaluation()


if __name__ == "__main__":
    main()