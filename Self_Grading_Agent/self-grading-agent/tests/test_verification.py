import pytest
from pydantic import ValidationError

from app.models.verification import VerificationResult


def make_result(**overrides):
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
        "reason": "Directly supported by the evidence.",
    }
    data.update(overrides)
    return VerificationResult(**data)


def test_valid_verification_result():
    result = make_result()

    assert result.supported is True
    assert result.support_score == 0.95
    assert result.evidence_sufficient is True


def test_support_score_cannot_exceed_one():
    with pytest.raises(ValidationError):
        make_result(support_score=1.01)


def test_support_score_cannot_be_negative():
    with pytest.raises(ValidationError):
        make_result(support_score=-0.01)


def test_zero_support_score_is_valid():
    result = make_result(
        supported=False,
        support_score=0.0,
        evidence_sufficient=False,
    )

    assert result.support_score == 0.0


def test_one_support_score_is_valid():
    result = make_result(support_score=1.0)

    assert result.support_score == 1.0


def test_all_verification_signals_are_present():
    result = make_result(
        entity_match=False,
        temporal_match=False,
        numerical_match=False,
        condition_match=False,
        contradiction=True,
        evidence_sufficient=True,
        unsupported_inference=True,
    )

    assert result.entity_match is False
    assert result.temporal_match is False
    assert result.numerical_match is False
    assert result.condition_match is False
    assert result.contradiction is True
    assert result.evidence_sufficient is True
    assert result.unsupported_inference is True


def test_reason_is_stored():
    result = make_result(
        reason="The requested date is not present in the evidence."
    )

    assert "requested date" in result.reason
