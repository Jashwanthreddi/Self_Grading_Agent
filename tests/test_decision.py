from app.decision.confidence import (
    ConfidenceDecision,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    NO_ANSWER,
)
from app.models.verification import VerificationResult


def make_verification(**overrides):
    data = {
        "supported": True,
        "support_score": 0.95,
        "entity_match": True,
        "temporal_match": True,
        "numerical_match": True,
        "condition_match": True,
        "contradiction": False,
        "evidence_sufficient": True,
        "unsupported_inference": False,
        "reason": "The answer is directly supported by the evidence.",
    }
    data.update(overrides)
    return VerificationResult(**data)


def test_high_confidence_when_evidence_is_strong():
    decision = ConfidenceDecision()

    result = decision.decide(make_verification())

    assert result == HIGH_CONFIDENCE


def test_high_confidence_at_exact_threshold():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(support_score=0.80)
    )

    assert result == HIGH_CONFIDENCE


def test_low_confidence_below_threshold():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(support_score=0.79)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_entity_does_not_match():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(entity_match=False)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_temporal_match_fails():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(temporal_match=False)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_numerical_match_fails():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(numerical_match=False)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_condition_match_fails():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(condition_match=False)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_contradiction_exists():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(contradiction=True)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_unsupported_inference_exists():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(unsupported_inference=True)
    )

    assert result == LOW_CONFIDENCE


def test_low_confidence_when_verifier_does_not_support_answer():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(supported=False)
    )

    assert result == LOW_CONFIDENCE


def test_no_answer_when_evidence_is_insufficient():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(
            evidence_sufficient=False,
            supported=False,
            support_score=0.0,
        )
    )

    assert result == NO_ANSWER


def test_no_answer_has_priority_over_other_failures():
    decision = ConfidenceDecision()

    result = decision.decide(
        make_verification(
            evidence_sufficient=False,
            contradiction=True,
            entity_match=False,
            temporal_match=False,
            numerical_match=False,
            condition_match=False,
            unsupported_inference=True,
            supported=False,
            support_score=0.1,
        )
    )

    assert result == NO_ANSWER
