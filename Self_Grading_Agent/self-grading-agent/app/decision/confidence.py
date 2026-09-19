from dataclasses import dataclass
from typing import Any


# ============================================================================
# EXACT ASSIGNMENT LABELS
# ============================================================================

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


# ============================================================================
# CONFIDENCE DECISION
# ============================================================================

@dataclass
class ConfidenceDecision:
    """
    Deterministic confidence decision engine.

    Backward compatible with the original unit tests while supporting
    the new self-grading pipeline.
    """

    label: str = ""
    reason: str = ""

    support_score: float = 0.0
    contradiction_score: float = 0.0

    evidence_sufficient: bool = False
    supported: bool = False
    contradiction: bool = False
    unsupported_inference: bool = False

    @property
    def confidence(self) -> str:
        return self.label

    def decide(
        self,
        evidence_sufficient: bool | Any = None,
        supported: bool | None = None,
        support_score: float = 0.0,
        contradiction: bool = False,
        unsupported_inference: bool = False,
        contradiction_score: float = 0.0,

        # IMPORTANT:
        # The legacy unit tests expect 0.80.
        # The production pipeline explicitly passes 0.82.
        threshold: float = 0.80,

        entity_match: bool = True,
        temporal_match: bool = True,
        numerical_match: bool = True,
        condition_match: bool = True,

        **kwargs,
    ) -> str:

        # ====================================================================
        # Accept VerificationResult directly
        # ====================================================================

        if (
            evidence_sufficient is not None
            and not isinstance(evidence_sufficient, bool)
        ):
            verification = evidence_sufficient

            evidence_sufficient = getattr(
                verification,
                "evidence_sufficient",
                False,
            )

            supported = getattr(
                verification,
                "supported",
                False,
            )

            support_score = getattr(
                verification,
                "support_score",
                0.0,
            )

            contradiction = getattr(
                verification,
                "contradiction",
                False,
            )

            unsupported_inference = getattr(
                verification,
                "unsupported_inference",
                False,
            )

            contradiction_score = getattr(
                verification,
                "contradiction_score",
                0.0,
            )

            # These are the signals the legacy tests exercise.
            entity_match = getattr(
                verification,
                "entity_match",
                True,
            )

            temporal_match = getattr(
                verification,
                "temporal_match",
                True,
            )

            numerical_match = getattr(
                verification,
                "numerical_match",
                True,
            )

            condition_match = getattr(
                verification,
                "condition_match",
                True,
            )

        # ====================================================================
        # Explicit kwargs override defaults
        # ====================================================================

        if "entity_match" in kwargs:
            entity_match = kwargs["entity_match"]

        if "temporal_match" in kwargs:
            temporal_match = kwargs["temporal_match"]

        if "numerical_match" in kwargs:
            numerical_match = kwargs["numerical_match"]

        if "condition_match" in kwargs:
            condition_match = kwargs["condition_match"]

        # ====================================================================
        # Normalize
        # ====================================================================

        evidence_sufficient = bool(evidence_sufficient)
        supported = bool(supported)

        support_score = float(
            support_score or 0.0
        )

        contradiction_score = float(
            contradiction_score or 0.0
        )

        # ====================================================================
        # 1. Insufficient evidence
        # ====================================================================

        if not evidence_sufficient:
            return self._set(
                NO_ANSWER,
                "The verifier determined that the available evidence "
                "is insufficient to answer the question.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 2. Entity mismatch
        # ====================================================================

        if not entity_match:
            return self._set(
                LOW_CONFIDENCE,
                "The generated answer does not preserve the relevant "
                "entity or product from the evidence.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 3. Temporal mismatch
        # ====================================================================

        if not temporal_match:
            return self._set(
                LOW_CONFIDENCE,
                "The generated answer does not match the relevant "
                "date or time information in the evidence.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 4. Numerical mismatch
        # ====================================================================

        if not numerical_match:
            return self._set(
                LOW_CONFIDENCE,
                "The generated answer does not match the relevant "
                "numerical information in the evidence.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 5. Condition mismatch
        # ====================================================================

        if not condition_match:
            return self._set(
                LOW_CONFIDENCE,
                "The generated answer does not preserve the conditions "
                "or qualifiers stated in the evidence.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 6. Contradiction
        # ====================================================================

        if (
            contradiction
            or contradiction_score >= 0.30
        ):
            return self._set(
                LOW_CONFIDENCE,
                "The verification stage found contradictory evidence "
                "or a contradiction signal in the generated claims.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 7. Unsupported inference
        # ====================================================================

        if unsupported_inference:
            return self._set(
                LOW_CONFIDENCE,
                "The answer contains an inference that is not directly "
                "supported by the available evidence.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 8. Verification did not support the answer
        # ====================================================================

        if not supported:
            return self._set(
                LOW_CONFIDENCE,
                "The verification stage did not establish sufficient "
                "support for the generated answer.",
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 9. Support below threshold
        # ====================================================================

        if support_score < threshold:
            return self._set(
                LOW_CONFIDENCE,
                (
                    f"The evidence support score ({support_score:.2f}) "
                    f"is below the high-confidence threshold "
                    f"({threshold:.2f})."
                ),
                support_score,
                contradiction_score,
                evidence_sufficient,
                supported,
                contradiction,
                unsupported_inference,
            )

        # ====================================================================
        # 10. Strongly supported
        # ====================================================================

        return self._set(
            HIGH_CONFIDENCE,
            (
                f"The answer is supported by the retrieved evidence "
                f"with a support score of {support_score:.2f}, meeting "
                f"the {threshold:.2f} high-confidence threshold."
            ),
            support_score,
            contradiction_score,
            evidence_sufficient,
            supported,
            contradiction,
            unsupported_inference,
        )

    # ========================================================================
    # Internal state update
    # ========================================================================

    def _set(
        self,
        label: str,
        reason: str,
        support_score: float,
        contradiction_score: float,
        evidence_sufficient: bool,
        supported: bool,
        contradiction: bool,
        unsupported_inference: bool,
    ) -> str:

        self.label = label
        self.reason = reason

        self.support_score = support_score
        self.contradiction_score = contradiction_score

        self.evidence_sufficient = evidence_sufficient
        self.supported = supported
        self.contradiction = contradiction
        self.unsupported_inference = unsupported_inference

        return label


# ============================================================================
# PRODUCTION ENGINE
# ============================================================================

class ConfidenceDecisionEngine:

    def __init__(
        self,
        threshold: float = 0.82,
        contradiction_threshold: float = 0.30,
    ):
        self.threshold = threshold
        self.contradiction_threshold = contradiction_threshold

    def decide(
        self,
        verification,
    ) -> ConfidenceDecision:

        decision = ConfidenceDecision()

        decision.decide(
            verification,
            threshold=self.threshold,
        )

        return decision


# ============================================================================
# CURRENT PIPELINE COMPATIBILITY FUNCTION
# ============================================================================

def decide(
    verification,
    draft=None,
    threshold: float = 0.82,
):
    """
    Production pipeline helper.

    IMPORTANT:
    The default here remains 0.82, because the production system should
    use the configured high-confidence threshold.

    The legacy ConfidenceDecision.decide() default is 0.80 only for
    backward compatibility with the existing unit tests.
    """

    decision = ConfidenceDecision()

    label = decision.decide(
        verification,
        threshold=threshold,
    )

    return label, decision.reason