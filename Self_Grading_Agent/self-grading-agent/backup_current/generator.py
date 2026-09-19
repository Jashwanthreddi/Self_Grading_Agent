from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from app.config import get_settings
from app.models.schemas import DraftAnswer, RetrievedPassage


GENERATION_SYSTEM_PROMPT = """
You are the answer-generation component of a retrieval-augmented
question-answering system.

Your task is to answer the user's question using ONLY the supplied evidence.

Rules:

1. Use only information contained in the evidence.
2. Never use outside knowledge.
3. Never guess.
4. Never invent missing facts.
5. Preserve exact entities, dates, numbers, products, organizations,
   and conditions from the evidence.
6. If the evidence does not contain enough information to answer the
   question, explicitly answer that the information is not available
   in the provided documents.
7. Select only the document IDs that actually support the answer.
8. Keep the answer concise and factual.
"""


class AnswerGenerator:

    def __init__(self):
        settings = get_settings()

        base_llm = ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.llm_model,
            temperature=0,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

        self.llm = base_llm.with_structured_output(
            DraftAnswer,
            method="json_schema",
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERATION_SYSTEM_PROMPT),
                (
                    "human",
                    """
Question:
{question}

Retrieved evidence:

{evidence}
""",
                ),
            ]
        )

        self.chain = self.prompt | self.llm

    @staticmethod
    def _format_evidence(
        passages: list[RetrievedPassage],
    ) -> str:

        if not passages:
            return "No evidence was retrieved."

        blocks = []

        for passage in passages:
            blocks.append(
                f"""
[Document ID: {passage.document_id}]
Document: {passage.document_name}

{passage.content}
"""
            )

        return "\n---\n".join(blocks)

    def generate(
        self,
        question: str,
        passages: list[RetrievedPassage],
    ) -> DraftAnswer:

        if not question.strip():
            raise ValueError("Question cannot be empty.")

        if not passages:
            return DraftAnswer(
                answer="I don't know based on the provided documents.",
                evidence_ids=[],
            )

        result = self.chain.invoke(
            {
                "question": question,
                "evidence": self._format_evidence(passages),
            }
        )

        if not isinstance(result, DraftAnswer):
            raise RuntimeError(
                "Generator returned an unexpected structured output."
            )

        return result