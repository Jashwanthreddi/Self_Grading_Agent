from __future__ import annotations

import re

from sentence_transformers import CrossEncoder

from app.models.schemas import (
    ClaimCheck,
    DraftAnswer,
    RetrievedPassage,
    VerificationResult,
)


class NliVerifier:
    """
    Independent verification stage of the Self-Grading Agent.

    Pipeline:

        Draft Answer
            ↓
        Claim Verification
            ↓
        Evidence Support
            ↓
        Entity / Date / Number / Condition Checks
            ↓
        VerificationResult

    The verifier is intentionally separate from generation.
    It never generates or rewrites the answer.
    """

    def __init__(self, settings):
        self.settings = settings

        self.model = CrossEncoder(
            settings.nli_model
        )

        self.entailment_threshold = (
            settings.nli_entailment_threshold
        )

        self.contradiction_threshold = (
            settings.nli_contradiction_threshold
        )

    # ================================================================
    # PUBLIC API
    # ================================================================

    def verify(
        self,
        draft: DraftAnswer,
        passages: list[RetrievedPassage],
    ) -> VerificationResult:
        """
        Independently verify the generated answer against
        the retrieved evidence.
        """

        # ------------------------------------------------------------
        # No answer generated
        # ------------------------------------------------------------

        if draft.abstain:
            return VerificationResult(
                supported=False,
                evidence_sufficient=False,
                contradiction=False,
                unsupported_inference=False,
                support_score=0.0,
                contradiction_score=0.0,
                claim_results=[],
                reason=(
                    "The generation stage abstained because the "
                    "available evidence was insufficient."
                ),
                method="Local NLI + deterministic consistency checks",
            )

        # ------------------------------------------------------------
        # No evidence
        # ------------------------------------------------------------

        if not passages:
            return VerificationResult(
                supported=False,
                evidence_sufficient=False,
                contradiction=False,
                unsupported_inference=True,
                support_score=0.0,
                contradiction_score=0.0,
                claim_results=[],
                reason=(
                    "No evidence passages were retrieved for "
                    "verification."
                ),
                method="Local NLI + deterministic consistency checks",
            )

        # ------------------------------------------------------------
        # Extract claims
        # ------------------------------------------------------------

        claims = [
            claim.strip()
            for claim in draft.claims
            if claim and claim.strip()
        ]

        # Fallback: verify the answer itself if the generator
        # did not return explicit claims.
        if not claims and draft.answer.strip():
            claims = [draft.answer.strip()]

        if not claims:
            return VerificationResult(
                supported=False,
                evidence_sufficient=False,
                contradiction=False,
                unsupported_inference=True,
                support_score=0.0,
                contradiction_score=0.0,
                claim_results=[],
                reason=(
                    "The generation stage did not produce a "
                    "verifiable answer or factual claim."
                ),
                method="Local NLI + deterministic consistency checks",
            )

        # ------------------------------------------------------------
        # Verify every claim independently
        # ------------------------------------------------------------

        claim_results: list[ClaimCheck] = []

        for claim in claims:
            result = self._verify_claim(
                claim=claim,
                passages=passages,
            )
            claim_results.append(result)

        # ------------------------------------------------------------
        # Aggregate claim-level results
        # ------------------------------------------------------------

        support_score = min(
            result.entailment
            for result in claim_results
        )

        contradiction_score = max(
            result.contradiction
            for result in claim_results
        )

        has_contradicted_claim = any(
            result.verdict == "contradicted"
            for result in claim_results
        )

        has_uncertain_claim = any(
            result.verdict == "uncertain"
            for result in claim_results
        )

        all_claims_supported = all(
            result.verdict == "supported"
            for result in claim_results
        )

        # ------------------------------------------------------------
        # Select only evidence relevant to the generated claims
        # ------------------------------------------------------------

        relevant_passages = self._select_relevant_passages(
            claim_results,
            passages,
        )

        relevant_evidence = "\n".join(
            passage.text
            for passage in relevant_passages
        )

        # ------------------------------------------------------------
        # Deterministic consistency checks
        # ------------------------------------------------------------

        entity_match = self._check_entities(
            draft.answer,
            relevant_evidence,
        )

        temporal_match = self._check_temporal_values(
            draft.answer,
            relevant_evidence,
        )

        numerical_match = self._check_numerical_values(
            draft.answer,
            relevant_evidence,
        )

        condition_match = self._check_conditions(
            draft.answer,
            relevant_evidence,
        )

        # ------------------------------------------------------------
        # Contradiction
        # ------------------------------------------------------------

        contradiction = has_contradicted_claim

        if (
            not contradiction
            and contradiction_score >= self.contradiction_threshold
            and support_score < self.entailment_threshold
        ):
            contradiction = True

        # ------------------------------------------------------------
        # Evidence sufficiency
        # ------------------------------------------------------------

        evidence_sufficient = bool(
            all_claims_supported
            and support_score >= self.entailment_threshold
        )

        # ------------------------------------------------------------
        # Unsupported inference
        # ------------------------------------------------------------

        unsupported_inference = bool(
            has_uncertain_claim
            or not entity_match
            or not temporal_match
            or not numerical_match
            or not condition_match
        )

        # ------------------------------------------------------------
        # Overall verification result
        # ------------------------------------------------------------

        supported = bool(
            evidence_sufficient
            and not contradiction
            and not unsupported_inference
        )

        reason = self._build_reason(
            supported=supported,
            evidence_sufficient=evidence_sufficient,
            contradiction=contradiction,
            unsupported_inference=unsupported_inference,
            support_score=support_score,
            contradiction_score=contradiction_score,
            entity_match=entity_match,
            temporal_match=temporal_match,
            numerical_match=numerical_match,
            condition_match=condition_match,
        )

        return VerificationResult(
            supported=supported,
            evidence_sufficient=evidence_sufficient,
            contradiction=contradiction,
            unsupported_inference=unsupported_inference,
            support_score=round(
                float(support_score),
                4,
            ),
            contradiction_score=round(
                float(contradiction_score),
                4,
            ),
            claim_results=claim_results,
            reason=reason,
            method=(
                "Local NLI CrossEncoder + "
                "deterministic entity/date/number/condition checks"
            ),
            entity_match=entity_match,
            temporal_match=temporal_match,
            numerical_match=numerical_match,
            condition_match=condition_match,
        )

    # ================================================================
    # CLAIM VERIFICATION
    # ================================================================

    def _verify_claim(
        self,
        claim: str,
        passages: list[RetrievedPassage],
    ) -> ClaimCheck:
        """
        Verify one generated claim against retrieved evidence.

        The retrieved passage can contain many unrelated facts.
        Before NLI evaluation, it is reduced to focused evidence
        sentences relevant to the claim.

        NLI format:

            premise   = focused evidence sentence
            hypothesis = generated claim

        Label mapping for the selected CrossEncoder:

            0 = contradiction
            1 = entailment
            2 = neutral
        """

        # ------------------------------------------------------------
        # Build focused evidence units
        # ------------------------------------------------------------

        evidence_units: list[tuple[RetrievedPassage, str]] = []

        for passage in passages:
            units = self._extract_relevant_evidence(
                claim,
                passage.text,
            )

            for unit in units:
                evidence_units.append(
                    (passage, unit)
                )

        # Safety fallback: if sentence/line extraction does not
        # identify anything, use the complete passage.
        if not evidence_units:
            for passage in passages:
                evidence_units.append(
                    (passage, passage.text)
                )

        # ------------------------------------------------------------
        # Run NLI
        # ------------------------------------------------------------

        pairs = [
            [evidence_text, claim]
            for _, evidence_text in evidence_units
        ]

        raw_scores = self.model.predict(
            pairs,
            apply_softmax=True,
        )

        best_entailment = 0.0
        best_contradiction = 0.0
        best_neutral = 0.0
        best_evidence_id: str | None = None

        passage_scores = []

        for (
            (passage, evidence_text),
            scores,
        ) in zip(
            evidence_units,
            raw_scores,
        ):
            # Confirmed model mapping:
            # 0 = contradiction
            # 1 = entailment
            # 2 = neutral

            contradiction = float(scores[0])
            entailment = float(scores[1])
            neutral = float(scores[2])

            passage_scores.append(
                (
                    passage,
                    evidence_text,
                    entailment,
                    contradiction,
                    neutral,
                )
            )

            if entailment > best_entailment:
                best_entailment = entailment
                best_evidence_id = passage.passage_id
                best_neutral = neutral
                best_contradiction = contradiction

        # ------------------------------------------------------------
        # Safety fallback
        # ------------------------------------------------------------

        if best_evidence_id is None and passage_scores:
            (
                best_passage,
                _,
                best_entailment,
                best_contradiction,
                best_neutral,
            ) = max(
                passage_scores,
                key=lambda item: item[2],
            )

            best_evidence_id = best_passage.passage_id

        # ------------------------------------------------------------
        # Claim verdict
        # ------------------------------------------------------------

        if best_entailment >= self.entailment_threshold:
            verdict = "supported"

        elif (
            best_contradiction >= self.contradiction_threshold
            and best_contradiction > best_entailment
        ):
            verdict = "contradicted"

        else:
            verdict = "uncertain"

        return ClaimCheck(
            claim=claim,
            entailment=round(
                best_entailment,
                4,
            ),
            contradiction=round(
                best_contradiction,
                4,
            ),
            neutral=round(
                best_neutral,
                4,
            ),
            best_evidence_id=best_evidence_id,
            verdict=verdict,
        )

    # ================================================================
    # FOCUSED EVIDENCE EXTRACTION
    # ================================================================

    @staticmethod
    def _extract_relevant_evidence(
        claim: str,
        passage_text: str,
    ) -> list[str]:
        """
        Extract focused evidence lines from a retrieved passage.

        Retrieval gives us document chunks. NLI verification works
        better when the premise is the smallest relevant factual
        statement rather than an entire multi-company document.

        This remains deterministic and does not introduce another
        model or another retrieval architecture.
        """

        # ------------------------------------------------------------
        # Normalize claim tokens
        # ------------------------------------------------------------

        claim_tokens = {
            token.lower()
            for token in re.findall(
                r"[A-Za-z0-9$€£₹.%/-]+",
                claim,
            )
            if len(token) >= 3
        }

        if not claim_tokens:
            return []

        # ------------------------------------------------------------
        # Split document into factual lines with section headers
        # ------------------------------------------------------------

        lines = []
        current_section = ""

        for raw_line in passage_text.splitlines():
            line = raw_line.strip()

            if not line or line.startswith("Document ID:") or line.startswith("Document Type:") or line.startswith("Last Updated:"):
                continue

            if line.startswith("#"):
                clean_heading = line.lstrip("#").strip()
                if "—" in clean_heading:
                    current_section = clean_heading.split("—")[0].strip()
                else:
                    current_section = clean_heading
                continue

            if not line.startswith("-") and not line.startswith("*") and ":" in line and len(line.split()) <= 5:
                current_section = line.rstrip(":")
                continue

            clean_line = re.sub(
                r"^[-*]\s*",
                "",
                line,
            ).strip()

            if not clean_line:
                continue

            if current_section and not clean_line.lower().startswith(current_section.lower()):
                context_line = f"{current_section}: {clean_line}"
            else:
                context_line = clean_line

            lines.append(context_line)

        # Also include raw whole passage and paragraphs as candidate units
        paragraphs = [p.strip() for p in passage_text.split("\n\n") if p.strip() and not p.strip().startswith("#")]
        for para in paragraphs:
            clean_para = re.sub(r"^[-*]\s*", "", para).strip()
            if clean_para and clean_para not in lines:
                lines.append(clean_para)

        if not lines:
            return []

        # ------------------------------------------------------------
        # Score each line using lexical overlap.
        # ------------------------------------------------------------

        scored_lines: list[tuple[float, str]] = []

        for line in lines:
            line_tokens = {
                token.lower()
                for token in re.findall(
                    r"[A-Za-z0-9$€£₹.%/-]+",
                    line,
                )
                if len(token) >= 3
            }

            overlap = claim_tokens.intersection(
                line_tokens
            )

            if not overlap:
                continue

            important_overlap = 0

            for token in overlap:
                if (
                    any(char.isdigit() for char in token)
                    or token[:1].isupper()
                    or token in {
                        "24/7",
                        "minimum",
                        "enterprise",
                        "premium",
                        "standard",
                        "launch",
                        "launched",
                        "support",
                        "month",
                        "user",
                        "workspace",
                    }
                ):
                    important_overlap += 1

            score = (
                len(overlap)
                + (important_overlap * 2.0)
            )

            scored_lines.append(
                (score, line)
            )

        if not scored_lines:
            return [lines[0]] if lines else []

        # Highest lexical matches first.
        scored_lines.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        selected = [
            line
            for _, line in scored_lines[:6]
        ]

        return selected

    # ================================================================
    # RELEVANT EVIDENCE SELECTION
    # ================================================================

    @staticmethod
    def _select_relevant_passages(
        claim_results: list[ClaimCheck],
        passages: list[RetrievedPassage],
    ) -> list[RetrievedPassage]:
        """
        Select evidence passages actually used to support the claims.

        This prevents unrelated retrieved documents from affecting
        entity/date/number/condition verification.
        """

        passage_map = {
            passage.passage_id: passage
            for passage in passages
        }

        selected: list[RetrievedPassage] = []
        seen: set[str] = set()

        for result in claim_results:
            evidence_id = result.best_evidence_id

            if (
                evidence_id
                and evidence_id in passage_map
                and evidence_id not in seen
            ):
                selected.append(
                    passage_map[evidence_id]
                )
                seen.add(evidence_id)

        # Fallback to retrieved passages if no best evidence was
        # selected by NLI.
        if not selected:
            return passages[:1]

        return selected

    # ================================================================
    # ENTITY CHECK
    # ================================================================

    def _check_entities(
        self,
        answer: str,
        evidence: str,
    ) -> bool:
        """
        Check important company, product and person names.

        This is an additional guard, not a replacement for NLI.
        """

        answer_entities = self._extract_entities(
            answer
        )

        if not answer_entities:
            return True

        evidence_lower = evidence.lower()

        for entity in answer_entities:
            if entity.lower() not in evidence_lower:
                return False

        return True

    @staticmethod
    def _extract_entities(
        text: str,
    ) -> set[str]:
        """
        Extract simple named entities and product-style names.

        Examples:

            NovaSearch
            OrbitBI
            RiskLens
            NexaTech Solutions
            Arjun Shah
        """

        entities: set[str] = set()

        patterns = [
            # Multi-word names.
            r"\b[A-Z][A-Za-z0-9&.-]+"
            r"(?:\s+[A-Z][A-Za-z0-9&.-]+)+\b",

            # CamelCase/product names.
            r"\b[A-Z][A-Za-z0-9]+"
            r"(?:[A-Z][A-Za-z0-9]+)+\b",
        ]

        for pattern in patterns:
            matches = re.findall(
                pattern,
                text,
            )

            for match in matches:
                value = match.strip()

                if len(value) >= 3:
                    entities.add(value)

        return entities

    # ================================================================
    # TEMPORAL CHECK
    # ================================================================

    def _check_temporal_values(
        self,
        answer: str,
        evidence: str,
    ) -> bool:
        """
        Check years and month/year combinations.
        """

        answer_dates = self._extract_temporal_values(
            answer
        )

        if not answer_dates:
            return True

        evidence_lower = evidence.lower()

        for value in answer_dates:
            if value.lower() not in evidence_lower:
                return False

        return True

    @staticmethod
    def _extract_temporal_values(
        text: str,
    ) -> set[str]:
        values: set[str] = set()

        years = re.findall(
            r"\b(?:19|20)\d{2}\b",
            text,
        )

        values.update(years)

        month_years = re.findall(
            r"\b(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)"
            r"\s+(?:19|20)\d{2}\b",
            text,
            flags=re.IGNORECASE,
        )

        values.update(
            value.strip()
            for value in month_years
        )

        return values

    # ================================================================
    # NUMERICAL CHECK
    # ================================================================

    def _check_numerical_values(
        self,
        answer: str,
        evidence: str,
    ) -> bool:
        """
        Verify numbers, percentages and monetary values.
        """

        answer_numbers = self._extract_numbers(
            answer
        )

        if not answer_numbers:
            return True

        evidence_numbers = self._extract_numbers(
            evidence
        )

        if not evidence_numbers:
            return False

        for number in answer_numbers:
            if number not in evidence_numbers:
                return False

        return True

    @staticmethod
    def _extract_numbers(
        text: str,
    ) -> set[str]:
        matches = re.findall(
            r"""
            (?:
                [$€£₹]\s*
            )?
            \d+(?:,\d{3})*
            (?:\.\d+)?
            %?
            """,
            text,
            flags=re.VERBOSE,
        )

        normalized: set[str] = set()

        for value in matches:
            value = value.replace(
                ",",
                "",
            ).strip()

            if value:
                normalized.add(value)

        return normalized

    # ================================================================
    # CONDITION CHECK
    # ================================================================

    def _check_conditions(
        self,
        answer: str,
        evidence: str,
    ) -> bool:
        """
        Check important qualifiers.

        Example:

            Evidence:
                24/7 support is available for Premium and Enterprise.

            Unsafe:
                24/7 support is available for all customers.

        The second statement removes an important condition.
        """

        evidence_lower = evidence.lower()
        answer_lower = answer.lower()

        # ------------------------------------------------------------
        # Premium / Enterprise qualification
        # ------------------------------------------------------------

        if (
            "24/7" in evidence_lower
            and (
                "premium" in evidence_lower
                or "enterprise" in evidence_lower
            )
        ):
            restricted_answer = (
                "premium" in answer_lower
                or "enterprise" in answer_lower
                or "plan" in answer_lower
            )

            universal_answer = any(
                phrase in answer_lower
                for phrase in [
                    "all customers",
                    "all users",
                    "every customer",
                    "everyone",
                    "all plans",
                    "regardless of plan",
                ]
            )

            if universal_answer and not restricted_answer:
                return False

        # ------------------------------------------------------------
        # Standard plan qualification
        # ------------------------------------------------------------

        if "standard" in evidence_lower:
            universal_answer = any(
                phrase in answer_lower
                for phrase in [
                    "all customers",
                    "all users",
                    "all plans",
                ]
            )

            if universal_answer:
                return False

        # ------------------------------------------------------------
        # Minimum seat/user requirement
        # ------------------------------------------------------------

        minimum_required = (
            "minimum" in evidence_lower
            or "at least" in evidence_lower
        )

        if minimum_required:
            answer_has_quantity = bool(
                re.search(
                    r"\b\d+\s*"
                    r"(?:seat|seats|user|users|workspace|workspaces)\b",
                    answer_lower,
                )
            )

            if (
                not answer_has_quantity
                and any(
                    word in answer_lower
                    for word in [
                        "all",
                        "any",
                        "no minimum",
                    ]
                )
            ):
                return False

        return True

    # ================================================================
    # VERIFICATION REASON
    # ================================================================

    @staticmethod
    def _build_reason(
        *,
        supported: bool,
        evidence_sufficient: bool,
        contradiction: bool,
        unsupported_inference: bool,
        support_score: float,
        contradiction_score: float,
        entity_match: bool,
        temporal_match: bool,
        numerical_match: bool,
        condition_match: bool,
    ) -> str:

        if contradiction:
            return (
                "The verifier detected a contradiction between "
                "the generated claim and the relevant evidence."
            )

        if not entity_match:
            return (
                "The generated answer introduced or changed an "
                "entity, company, or product not supported by the evidence."
            )

        if not temporal_match:
            return (
                "The generated answer does not match the relevant "
                "date or temporal information in the evidence."
            )

        if not numerical_match:
            return (
                "The generated answer does not match the relevant "
                "numerical information in the evidence."
            )

        if not condition_match:
            return (
                "The generated answer does not preserve an important "
                "condition or qualifier from the evidence."
            )

        if unsupported_inference:
            return (
                "At least one generated claim could not be directly "
                "supported by the retrieved evidence."
            )

        if not evidence_sufficient:
            return (
                "The retrieved evidence did not provide sufficiently "
                "strong entailment for all generated claims."
            )

        if supported:
            return (
                f"All generated claims were supported by the "
                f"relevant evidence. "
                f"Support={support_score:.2f}, "
                f"Contradiction={contradiction_score:.2f}."
            )

        return (
            "The verifier could not establish reliable support "
            "for the generated answer."
        )


# Backward-compatible name used by older code.
AnswerVerifier = NliVerifier