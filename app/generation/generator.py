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

        # Candidate models to try in sequence if primary experiences 503/429/404
        primary_model = self.settings.generation_model
        fallback_models = [
            primary_model,
            "gemini-2.5-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-flash-lite-latest",
            "gemini-3.7-flash",
        ]
        # Deduplicate while preserving order
        candidate_models = []
        for m in fallback_models:
            if m and m not in candidate_models:
                candidate_models.append(m)

        response = None
        last_exception = None

        for model_name in candidate_models:
            max_retries = 3
            base_delay = 1.5

            for attempt in range(max_retries):
                try:
                    response = self.client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=DraftAnswer,
                            max_output_tokens=700,
                            temperature=0.1,
                        ),
                    )
                    if response and response.text:
                        break
                except Exception as exc:
                    last_exception = exc
                    err_str = str(exc).lower()
                    is_transient = (
                        "503" in err_str
                        or "unavailable" in err_str
                        or "429" in err_str
                        or "resource_exhausted" in err_str
                        or "quota" in err_str
                        or "overloaded" in err_str
                    )

                    if is_transient and attempt < max_retries - 1:
                        import time
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        # Move on to next fallback model in candidate_models
                        break

            if response and response.text:
                break

        if response is None or not response.text:
            if last_exception:
                raise RuntimeError(
                    f"Gemini generation failed across candidate models: {last_exception}"
                ) from last_exception
            raise RuntimeError(
                "Gemini returned an empty generation response."
            )

        # Robust JSON extraction
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.removeprefix("```json").strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.removeprefix("```").strip()
        if raw_text.endswith("```"):
            raw_text = raw_text.removesuffix("```").strip()

        try:
            return DraftAnswer.model_validate_json(raw_text)
        except Exception:
            try:
                import json
                parsed = json.loads(raw_text)
                return DraftAnswer(
                    answer=str(parsed.get("answer", "")),
                    claims=list(parsed.get("claims", [])),
                    abstain=bool(parsed.get("abstain", False)),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Gemini returned a response that could not be parsed: {raw_text}"
                ) from exc


# Backward-compatible alias.
AnswerGenerator = GeminiGenerator