from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ============================================================================
# DOCUMENT
# ============================================================================

class Document(BaseModel):
    doc_id: str
    title: str
    source: str
    text: str


# ============================================================================
# RETRIEVED PASSAGE
# ============================================================================

class RetrievedPassage(BaseModel):
    """
    Canonical retrieval result.

    The model also supports the legacy fields used by the original
    unit tests:
        document_id
        document_name
        content
        score
        retrieval_method
    """

    # ------------------------------------------------------------------------
    # New canonical fields
    # ------------------------------------------------------------------------

    passage_id: str = ""
    doc_id: str = ""
    title: str = ""
    source: str = ""
    text: str = ""

    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0

    # Keep retrieval method explicitly so legacy tests can inspect it.
    retrieval_method: str = "hybrid"

    # ------------------------------------------------------------------------
    # Legacy -> canonical conversion
    # ------------------------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_fields(cls, values):
        if not isinstance(values, dict):
            return values

        values = dict(values)

        # document_id -> doc_id
        if not values.get("doc_id") and values.get("document_id"):
            values["doc_id"] = values["document_id"]

        # document_name -> title/source
        if not values.get("title") and values.get("document_name"):
            values["title"] = values["document_name"]

        if not values.get("source") and values.get("document_name"):
            values["source"] = values["document_name"]

        # content -> text
        if not values.get("text") and values.get("content"):
            values["text"] = values["content"]

        # score -> hybrid_score
        if (
            not values.get("hybrid_score")
            and values.get("score") is not None
        ):
            values["hybrid_score"] = values["score"]

        # Preserve legacy retrieval method.
        if values.get("retrieval_method"):
            values["retrieval_method"] = values["retrieval_method"]

        # Deterministic ID for old-style objects.
        if not values.get("passage_id"):
            doc_id = values.get("doc_id", "unknown")
            values["passage_id"] = f"{doc_id}-legacy"

        return values

    # ------------------------------------------------------------------------
    # Legacy compatibility properties
    # ------------------------------------------------------------------------

    @property
    def document_id(self) -> str:
        return self.doc_id

    @property
    def document_name(self) -> str:
        return self.title or self.source

    @property
    def content(self) -> str:
        return self.text

    @property
    def score(self) -> float:
        return self.hybrid_score


# ============================================================================
# DRAFT ANSWER
# ============================================================================

class DraftAnswer(BaseModel):
    answer: str = Field(
        description="Concise evidence-grounded answer."
    )

    claims: list[str] = Field(
        default_factory=list,
        description="Atomic factual claims made by the answer.",
    )

    abstain: bool = Field(
        default=False,
        description="True when evidence is insufficient.",
    )


# ============================================================================
# CLAIM VERIFICATION
# ============================================================================

class ClaimCheck(BaseModel):
    claim: str

    entailment: float
    contradiction: float
    neutral: float

    best_evidence_id: str | None = None

    verdict: Literal[
        "supported",
        "contradicted",
        "uncertain",
    ]


# ============================================================================
# VERIFICATION RESULT
# ============================================================================

class VerificationResult(BaseModel):
    supported: bool

    evidence_sufficient: bool

    contradiction: bool

    unsupported_inference: bool

    support_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    contradiction_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    claim_results: list[ClaimCheck] = Field(
        default_factory=list
    )

    reason: str

    method: str

    # ------------------------------------------------------------------------
    # Legacy verification signals
    #
    # These are useful deterministic signals for the confidence engine.
    # The new verifier can leave them at True when no mismatch is detected.
    # ------------------------------------------------------------------------

    entity_match: bool = True
    temporal_match: bool = True
    numerical_match: bool = True
    condition_match: bool = True