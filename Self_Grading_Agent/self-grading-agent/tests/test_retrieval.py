from app.models.schemas import RetrievedPassage


def test_retrieved_passage_model():
    passage = RetrievedPassage(
        document_id="KB-001",
        document_name="01_nexatech.md",
        content="NovaSearch launched in March 2025.",
        score=0.95,
        retrieval_method="hybrid",
    )

    assert passage.document_id == "KB-001"
    assert passage.document_name == "01_nexatech.md"
    assert passage.score == 0.95
    assert passage.retrieval_method == "hybrid"


def test_retrieved_passage_preserves_evidence_content():
    content = "NovaSearch Enterprise requires a minimum of 25 seats."

    passage = RetrievedPassage(
        document_id="KB-001",
        document_name="01_nexatech.md",
        content=content,
        score=0.90,
        retrieval_method="dense",
    )

    assert passage.content == content
    assert "25 seats" in passage.content


def test_retrieved_passage_supports_bm25_method():
    passage = RetrievedPassage(
        document_id="KB-011",
        document_name="11_support_policies.md",
        content="Premium customers receive priority support.",
        score=0.80,
        retrieval_method="bm25",
    )

    assert passage.retrieval_method == "bm25"


def test_retrieved_passage_supports_hybrid_method():
    passage = RetrievedPassage(
        document_id="KB-008",
        document_name="08_pricing_matrix.md",
        content="OrbitBI Professional costs $29 per user per month.",
        score=0.92,
        retrieval_method="hybrid",
    )

    assert passage.retrieval_method == "hybrid"


def test_multiple_passages_can_represent_competing_evidence():
    passages = [
        RetrievedPassage(
            document_id="KB-001",
            document_name="01_nexatech.md",
            content="NovaSearch launched in March 2025.",
            score=0.95,
            retrieval_method="hybrid",
        ),
        RetrievedPassage(
            document_id="KB-009",
            document_name="09_release_history.md",
            content="NovaSearch SAML support was introduced in June 2025.",
            score=0.91,
            retrieval_method="hybrid",
        ),
    ]

    assert len(passages) == 2
    assert "March 2025" in passages[0].content
    assert "June 2025" in passages[1].content


def test_trap_evidence_can_be_represented_for_verification():
    passages = [
        RetrievedPassage(
            document_id="KB-001",
            document_name="01_nexatech.md",
            content="NovaSearch costs $49 per user per month.",
            score=0.94,
            retrieval_method="hybrid",
        ),
        RetrievedPassage(
            document_id="KB-008",
            document_name="08_pricing_matrix.md",
            content="FlowPilot costs $79 per workspace per month.",
            score=0.88,
            retrieval_method="hybrid",
        ),
    ]

    combined = " ".join(p.content for p in passages)

    assert "$49" in combined
    assert "$79" in combined
    assert "FlowPilot" in combined
