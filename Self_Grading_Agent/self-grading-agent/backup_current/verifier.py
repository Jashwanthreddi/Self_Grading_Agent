from langchain_groq import ChatGroq

from app.config import get_settings
from app.models.schemas import DraftAnswer, RetrievedPassage
from app.models.verification import VerificationResult


class AnswerVerifier:
    """
    Independent verification stage.

    The verifier does NOT decide whether the generated answer merely
    sounds plausible. It checks the claims in the draft against the
    retrieved evidence.
    """

    def __init__(self):
        settings = get_settings()

        self.llm = ChatGroq(
            model=settings.llm_model,
            temperature=0,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        ).with_structured_output(
            VerificationResult,
            method="json_schema",
        )

    def _format_evidence(
        self,
        passages: list[RetrievedPassage],
    ) -> str:
        if not passages:
            return "NO EVIDENCE WAS RETRIEVED."

        blocks = []

        for passage in passages:
            blocks.append(
                f"[{passage.document_id}] {passage.document_name}\n"
                f"{passage.content}"
            )

        return "\n\n".join(blocks)

    def verify(
        self,
        question: str,
        draft: DraftAnswer,
        passages: list[RetrievedPassage],
    ) -> VerificationResult:

        if not passages:
            return VerificationResult(
                supported=False,
                support_score=0.0,
                entity_match=False,
                temporal_match=False,
                numerical_match=False,
                condition_match=False,
                contradiction=False,
                evidence_sufficient=False,
                unsupported_inference=True,
                reason=(
                    "No relevant evidence was retrieved, so the answer "
                    "cannot be verified from the knowledge base."
                ),
            )

        evidence = self._format_evidence(passages)

        prompt = f"""
You are an independent answer verifier.

Your job is to verify a generated answer against the supplied evidence.

Do NOT assume the draft answer is correct.
Do NOT use outside knowledge.
Do NOT reward an answer merely because it sounds plausible.

QUESTION:
{question}

DRAFT ANSWER:
{draft.answer}

RETRIEVED EVIDENCE:
{evidence}

Verification rules:

1. ENTITY MATCH
Check that the company, product, person, organization, location,
or other important entity in the draft matches the entity asked about.

2. TEMPORAL MATCH
Check dates, years, launch dates, release dates, and time periods.
A date belonging to a different event is NOT a match.

3. NUMERICAL MATCH
Check prices, percentages, quantities, seat counts, availability
targets, and other numerical claims exactly.

4. CONDITION MATCH
Check qualifiers such as:
- standard vs premium
- enterprise vs all customers
- optional vs guaranteed
- customer type
- plan restrictions
- region restrictions
- "only", "all", "minimum", "maximum", etc.

5. CONTRADICTION
Set contradiction=true when the evidence conflicts with the
claim made by the draft answer.

IMPORTANT:
If the evidence says the requested claim is false, the draft must
NOT be considered supported.

For example:
Question: "Does Product A cost $79?"
Evidence: "Product A costs $49."
Then:
contradiction=true
supported=false

6. EVIDENCE SUFFICIENCY
Set evidence_sufficient=true ONLY when the retrieved evidence
contains enough information to answer the question reliably.

If the requested fact is absent, set:
evidence_sufficient=false

Examples:
- Question asks for 2026 revenue, but no 2026 revenue exists in evidence
  -> evidence_sufficient=false
- Question asks for number of employees, but documents give no employee count
  -> evidence_sufficient=false
- Question asks for a programming language, but documents don't state one
  -> evidence_sufficient=false

7. AMBIGUITY
If the evidence provides related information but does not resolve
the exact question, do not mark it fully supported.

Example:
Question: "Is Singapore the main market?"
Evidence: "Markets: India and Singapore."
The evidence does NOT establish which is the main market.
Therefore:
evidence_sufficient=false OR supported=false
unsupported_inference=true

8. NEGATIVE QUESTIONS
A negative answer can be supported if the evidence explicitly
contradicts the claim.

Example:
Question: "Did NovaSearch launch in June 2025?"
Evidence: "NovaSearch launched in March 2025."
The answer "No" is supported because the evidence directly
establishes March 2025.

9. MISSING INFORMATION
Do not convert absence of information into a positive factual claim.

10. SUPPORT SCORE
Use a score from 0 to 1:
- 0.95-1.00: directly and unambiguously supported
- 0.80-0.94: strongly supported with minor limitations
- 0.50-0.79: partial, ambiguous, or qualified support
- below 0.50: weak or unsupported

Final consistency rules:

- contradiction=true -> supported MUST be false
- evidence_sufficient=false -> supported MUST be false
- unsupported_inference=true -> supported should normally be false
- If a required qualifier is missing, condition_match=false
- If the requested entity/date/number does not match, the relevant
  match field must be false

Return only the structured verification result.
"""

        result = self.llm.invoke(prompt)

        # Defensive consistency enforcement.
        # These rules prevent an internally inconsistent LLM result
        # from being converted into high confidence.
        if result.contradiction:
            result.supported = False

        if not result.evidence_sufficient:
            result.supported = False

        if result.unsupported_inference:
            result.supported = False

        if not result.entity_match:
            result.supported = False

        if not result.temporal_match:
            result.supported = False

        if not result.numerical_match:
            result.supported = False

        if not result.condition_match:
            result.supported = False

        return result
