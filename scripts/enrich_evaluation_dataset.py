import json
from pathlib import Path


DATASET_PATH = Path("data/evaluation/test_questions.json")


ANCHORS = {
    "Q001": {
        "required_keywords": ["NovaSearch", "March 2025"],
    },
    "Q002": {
        "required_keywords": ["Arjun Shah"],
    },
    "Q003": {
        "required_keywords": ["Frankfurt"],
    },
    "Q004": {
        "required_keywords": ["$29", "user", "month"],
    },
    "Q005": {
        "required_keywords": ["solar", "asset", "monitoring"],
    },
    "Q006": {
        "required_keywords": ["99.95%"],
    },
    "Q007": {
        "required_keywords": ["CEO"],
    },
    "Q008": {
        "required_keywords": ["AES-256"],
    },
    "Q009": {
        "required_keywords": ["RiskLens"],
    },
    "Q010": {
        "required_keywords": ["25", "seats"],
    },

    # Unanswerable questions
    "Q011": {
        "required_keywords": [],
    },
    "Q012": {
        "required_keywords": [],
    },
    "Q013": {
        "required_keywords": [],
    },
    "Q014": {
        "required_keywords": [],
    },
    "Q015": {
        "required_keywords": [],
    },

    # Partially supported / ambiguous
    "Q016": {
        "required_keywords": ["24/7", "Premium"],
    },
    "Q017": {
        "required_keywords": ["priority"],
    },
    "Q018": {
        "required_keywords": [],
    },
    "Q019": {
        "required_keywords": ["India", "Singapore"],
    },
    "Q020": {
        "required_keywords": ["enterprise", "24/7"],
    },

    # Traps
    "Q021": {
        "required_keywords": [],
    },
    "Q022": {
        "required_keywords": ["April 2022"],
        "forbidden_facts": ["August 2023"],
    },
    "Q023": {
        "required_keywords": ["No", "FlowPilot"],
        "forbidden_facts": [
            "NovaSearch costs $79",
            "NovaSearch $79",
        ],
    },
    "Q024": {
        "required_keywords": ["No", "Mumbai"],
        "forbidden_facts": [
            "Frankfurt",
        ],
    },
    "Q025": {
        "required_keywords": ["No", "March 2025"],
        "forbidden_facts": [
            "NovaSearch launched in June 2025",
            "NovaSearch launch in June 2025",
        ],
    },
}


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATASET_PATH}"
        )

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        questions = json.load(file)

    if not isinstance(questions, list):
        raise ValueError(
            "Evaluation dataset must contain a JSON list."
        )

    question_ids = {
        item["id"]
        for item in questions
    }

    missing = set(ANCHORS) - question_ids

    if missing:
        raise ValueError(
            f"Dataset is missing expected question IDs: "
            f"{sorted(missing)}"
        )

    for item in questions:
        question_id = item["id"]

        item.update(
            ANCHORS[question_id]
        )

    with DATASET_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            questions,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Updated {len(questions)} evaluation questions."
    )

    print(
        f"Dataset: {DATASET_PATH}"
    )


if __name__ == "__main__":
    main()