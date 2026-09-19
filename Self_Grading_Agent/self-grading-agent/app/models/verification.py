from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Structured result produced by the independent verifier."""

    supported: bool
    support_score: float = Field(ge=0.0, le=1.0)

    entity_match: bool
    temporal_match: bool
    numerical_match: bool
    condition_match: bool

    contradiction: bool
    evidence_sufficient: bool
    unsupported_inference: bool

    reason: str