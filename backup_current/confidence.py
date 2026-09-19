from app.models.verification import VerificationResult


HIGH_CONFIDENCE = (
    "High confidence — the answer is clearly supported by the sources."
)

LOW_CONFIDENCE = (
    "Low confidence — the available evidence is incomplete, "
    "ambiguous, or requires an unsupported inference."
)

NO_ANSWER = (
    "I don't know — the documents do not contain enough evidence to answer."
)


class ConfidenceDecision:
    """
    Converts independent verification signals into one of the
    assignment's required confidence labels.

    The decision is deliberately conservative:
    missing evidence produces I don't know,
    while ambiguity or failed verification produces Low confidence.
    """

    def decide(self, verification: VerificationResult) -> str:

        # Missing evidence is stronger than every other signal.
        if not verification.evidence_sufficient:
            return NO_ANSWER

        # Any direct contradiction means the generated answer
        # cannot receive high confidence.
        if verification.contradiction:
            return LOW_CONFIDENCE

        # Required claim-level checks.
        if not verification.entity_match:
            return LOW_CONFIDENCE

        if not verification.temporal_match:
            return LOW_CONFIDENCE

        if not verification.numerical_match:
            return LOW_CONFIDENCE

        if not verification.condition_match:
            return LOW_CONFIDENCE

        # The answer requires an inference not explicitly supported.
        if verification.unsupported_inference:
            return LOW_CONFIDENCE

        # The verifier itself did not establish direct support.
        if not verification.supported:
            return LOW_CONFIDENCE

        # Conservative threshold.
        if verification.support_score < 0.80:
            return LOW_CONFIDENCE

        return HIGH_CONFIDENCE
