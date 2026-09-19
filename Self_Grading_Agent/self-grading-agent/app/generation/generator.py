from __future__ import annotations

from google import genai
from google.genai import types

from app.models.schemas import DraftAnswer


SYSTEM_INSTRUCTION = """
You are the GENERATION STAGE of a production-style Self-Grading Agent.

Your job is ONLY to generate a draft answer from the retrieved evidence.

STRICT RULES:

1. Use ONLY the supplied evidence.
2. Do NOT use outside knowledge.
3. Do NOT guess or invent missing information.
4. Preserve exact:
   - company names
   - person names
   - product names
   - dates
   - prices
   - percentages
   - quantities
   - locations
   - conditions
   - relationships
5. Trap questions may intentionally change one important entity, product,
   date, number, organization, or condition. Detect the difference using
   the evidence.
6. If the evidence is insufficient to answer the question, abstain.
7. If the question contains a false proposition, correct it using the evidence.
8. Break the generated answer into atomic factual claims.
9. Every claim must be directly supported by the supplied evidence.
10. Do not make assumptions from common knowledge.
11. Keep the answer concise and professional.
12. The verification stage will independently evaluate your claims.
13. Do NOT assign a confidence label yourself. Confidence is decided later
    by the independent verification and confidence-decision stages.

IMPORTANT:
You are not the final judge.
You are only responsible for producing a grounded draft answer and its claims.
"""


class GeminiGenerator:
    """
    Generation stage of the Self-Grading Agent.

    Responsibilities:
    - Consume retrieved evidence.
    - Generate an evidence-grounded draft answer.
    - Extract atomic claims.
    - Abstain when evidence is insufficient.

    Verification and confidence assignment are intentionally handled
    outside this class.
    """

    def __init__(self, settings):
        self.settings = settings

        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Add it to the project .env file."
            )

        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

    def generate(
        self,
        question: str,
        passages,
    ) -> DraftAnswer:
        """
        Generate a structured draft answer from retrieved evidence.
        """

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not passages:
            return DraftAnswer(
                answer="",
                claims=[],
                abstain=True,
            )

        evidence_blocks = []

        for passage in passages:
            evidence_blocks.append(
                f"""
[{passage.passage_id}]
Document: {passage.title}
Source: {passage.source}

{passage.text}
""".strip()
            )

        evidence = "\n\n".join(evidence_blocks)

        prompt = f"""
{SYSTEM_INSTRUCTION}

QUESTION:
{question.strip()}

RETRIEVED EVIDENCE:
{evidence}

TASK:
Generate the best evidence-grounded draft answer.

If the evidence clearly answers the question:
- provide a concise answer
- extract the atomic factual claims supporting that answer
- set abstain to false

If the evidence does not contain enough information:
- leave the answer concise or empty
- provide no unsupported claims
- set abstain to true

Return ONLY the structured response.
"""

        response = self.client.models.generate_content(
            model=self.settings.generation_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=DraftAnswer,
                max_output_tokens=700,
                temperature=0.1,
            ),
        )

        if not response.text:
            raise RuntimeError(
                "Gemini returned an empty generation response."
            )

        try:
            return DraftAnswer.model_validate_json(
                response.text
            )
        except Exception as exc:
            raise RuntimeError(
                "Gemini returned a response that could not be "
                "validated against DraftAnswer."
            ) from exc


# Backward-compatible alias.
AnswerGenerator = GeminiGenerator