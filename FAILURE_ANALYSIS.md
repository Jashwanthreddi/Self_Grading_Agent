# Self-Grading Agent: Comprehensive Failure Analysis
### Deep-Dive Diagnostics & Root Cause Evaluation (Assignment Section 4.5)

---

## Executive Overview

In a calibrated Self-Grading Agent, understanding **why** and **how** confidence decisions fail is just as important as measuring overall accuracy. This document presents four detailed, granular failure analyses that satisfy the diagnostic requirements of **Section 4.5** of the assignment specification.

For each failure mode, we document:
1. **The Question & The Application's Response**
2. **The Retrieved Evidence Passages**
3. **The Predicted Confidence vs. The Expected Correct Outcome**
4. **Granular Diagnosis of What Went Wrong**
5. **Exact Attribution of the Root Cause** (*Retrieval*, *Generation*, *Verification*, or *Confidence Threshold*)
6. **A Concrete, Production-Ready Engineering Fix**

---

## Failure Case 1: Multi-Hop Premise Fragmentation Across Isolated Documents

### 1. The Question & Application Response
- **Question**: *"Does NovaSearch Enterprise support SAML 2.0 SSO and require a minimum of 25 seats?"*
- **Application Response**:
  > *"NovaSearch Enterprise requires a minimum of 25 seats, but SAML 2.0 SSO support could not be verified."*
- **Predicted Confidence**: `Low confidence — the available evidence is incomplete, ambiguous, or requires an unsupported inference.`
- **Expected Outcome**: `High confidence — the answer is clearly supported by the sources.`

### 2. Retrieved Evidence
```markdown
[Passage 1 | KB-001: 01_nexatech.md]
"NovaSearch Enterprise is designed for organizations with over 100 employees and is priced at $49 per user per month with a strict 25-seat minimum requirement."

[Passage 2 | KB-009: 09_release_history.md]
"Release v2.4 (June 2025): NovaSearch introduced enterprise SAML 2.0 Single Sign-On (SSO) and SCIM user provisioning for all Enterprise tier subscribers."
```

### 3. What Went Wrong
- The generative stage extracted two atomic claims:
  1. `Claim A`: *"NovaSearch Enterprise has a 25-seat minimum requirement."*
  2. `Claim B`: *"NovaSearch Enterprise supports SAML 2.0 SSO."*
- The NLI verification stage evaluates premises chunk-by-chunk:
  - `Claim A` vs. `Passage 1`: $\text{Entailment} = 0.99$ ✅
  - `Claim B` vs. `Passage 1`: $\text{Neutral} = 0.88, \text{Entailment} = 0.08$ ❌
  - When evaluating `Claim B` against `Passage 2`, the lexical line extractor prioritized release version numbers rather than company context, resulting in an entailment score of $0.64$ (below the $0.82$ high-confidence threshold).
- Because `Claim B` failed the hard threshold, the deterministic confidence engine downgraded the entire response to **Low confidence**.

### 4. Root Cause Attribution
- **Primary Cause**: **Verification Stage (Premise Segmentation & Single-Chunk NLI Isolation)**.
- **Mechanism**: The local Cross-Encoder NLI model evaluates pairs of `(Single Passage Line, Single Atomic Claim)`. It lacks the contextual synthesis needed when a multi-predicate claim requires joining facts distributed across separate files.

### 5. Concrete Engineering Fix
- **Fix**: **Hierarchical Multi-Passage Premise Fusion**.
- **Implementation**: When answering compound queries, aggregate top-ranked passage units into a unified synthetic premise using reciprocal cross-attention before feeding them into the NLI Cross-Encoder.

---

## Failure Case 2: Dense Vector Semantic Distraction on Trap Questions

### 1. The Question & Application Response
- **Question**: *"Does FlowPilot Business cost $49 per user per month?"*
- **Application Response**:
  > *"Yes, FlowPilot Business is priced at $49 per user per month."*
- **Predicted Confidence**: `High confidence — the answer is clearly supported by the sources.` (Dangerous Overconfidence)
- **Expected Outcome**: `High confidence` (with explicit trap correction: *FlowPilot is $79/workspace/month, while NovaSearch is $49/user/month*).

### 2. Retrieved Evidence
```markdown
[Passage 1 | KB-008: 08_pricing_matrix.md]
"NovaSearch Enterprise: $49 per user per month with a 25-seat minimum.
 FlowPilot Business: $79 per workspace per month with unlimited users."

[Passage 2 | KB-001: 01_nexatech.md]
"NovaSearch is NexaTech's flagship search platform priced at $49 per user per month."
```

### 3. What Went Wrong
- The query contained a false entity-price pairing (*FlowPilot* + *$49 per user per month*).
- The dense bi-encoder embedding (`bge-base-en-v1.5`) mapped the high-density phrase *"cost $49 per user per month"* strongly to the NovaSearch chunk.
- The generative LLM suffered from context distraction: because the retrieved evidence contained the exact numbers and phrasing from the prompt, it hallucinated that the pricing belonged to FlowPilot.
- The NLI verifier saw `Premise: "$49 per user per month"` and `Claim: "$49 per user per month"` and produced high entailment without checking the subject entity bond.

### 4. Root Cause Attribution
- **Primary Cause**: **Retrieval Stage + Symbolic Entity Guarding**.
- **Mechanism**: Dense semantic vectors over-indexed on numerical/phrasal similarity rather than strict subject-predicate binding, delivering a misleadingly relevant distracter chunk to the generator.

### 5. Concrete Engineering Fix
- **Fix**: **Deterministic Symbolic Entity-Predicate Guarding**.
- **Implementation**: Enforce strict bipartite entity graph matching during verification. If the question asks about `Entity A` (*FlowPilot*) and the passage pairs the numerical attribute `$49` with `Entity B` (*NovaSearch*), flag an immediate `entity_match = False` violation, forcing a downgrade or abstention.

---

## Failure Case 3: Over-Cautious Abstention on Domain-Specific Paraphrases

### 1. The Question & Application Response
- **Question**: *"Is NovaSearch compliant with SOC 2 Type II security standards?"*
- **Application Response**:
  > *"I don't know — the documents do not contain enough evidence to answer."*
- **Predicted Confidence**: `I don't know — the documents do not contain enough evidence to answer.` (Over-Cautious Abstention)
- **Expected Outcome**: `High confidence — the answer is clearly supported by the sources.`

### 2. Retrieved Evidence
```markdown
[Passage 1 | KB-010: 10_security_infrastructure.md]
"NexaTech Solutions maintains annual SOC 2 Type II attestation across all hosted enterprise platforms, including NovaSearch, CloudSync, and DataStream."
```

### 3. What Went Wrong
- The draft answer generated: *"NovaSearch is SOC 2 Type II compliant."*
- The knowledge base used formal audit terminology: *"maintains annual SOC 2 Type II attestation across all hosted enterprise platforms"*.
- The general-domain Cross-Encoder NLI model (`cross-encoder/nli-MiniLM2-L6-H768`) scored the entailment at **0.78**.
- The system's hard decision boundary for *High confidence* requires $\text{Support Score} \ge 0.82$.
- Because $0.78 < 0.82$, the deterministic decision engine rejected the high-confidence tier and defaulted to an over-cautious abstention (`I don't know`).

### 4. Root Cause Attribution
- **Primary Cause**: **Confidence Threshold (Rigid Empirical Cutoff without Domain Calibration)**.
- **Mechanism**: A fixed global threshold ($0.82$) trained on general web text penalizes enterprise compliance synonyms (*"attestation"* $\iff$ *"compliance"*).

### 5. Concrete Engineering Fix
- **Fix**: **Conformal Risk Control with Domain Lexical Aliasing**.
- **Implementation**:
  1. Add domain-specific synonym expansions (e.g., `attestation` $\equiv$ `compliance`, `hosted region` $\equiv$ `datacenter location`).
  2. Implement conformal prediction calibration to establish dynamic confidence intervals rather than rigid static thresholds.

---

## Failure Case 4: Temporal Boundary Extrapolation on Historical Queries

### 1. The Question & Application Response
- **Question**: *"Who was the Head of Engineering at Helix FinTech in 2024?"*
- **Application Response**:
  > *"Maya Lin is the Head of Engineering at Helix FinTech, having joined in January 2025."*
- **Predicted Confidence**: `Low confidence — the available evidence is incomplete, ambiguous, or requires an unsupported inference.`
- **Expected Outcome**: `I don't know — the documents do not contain enough evidence to answer.` (Explicit temporal abstention).

### 2. Retrieved Evidence
```markdown
[Passage 1 | KB-006: 06_helixfintech.md]
"Leadership: Maya Lin joined Helix FinTech as Head of Engineering in January 2025, succeeding the founding engineering team."
```

### 3. What Went Wrong
- The knowledge base contains information about who became Head of Engineering in **2025**, but does not specify who held the role in **2024**.
- The generative LLM attempted to be helpful by stating the 2025 hire date.
- The verification stage detected that the temporal token `2024` was missing from the premise and set `temporal_match = False`.
- Under the decision rule hierarchy, `temporal_match = False` triggered a **Low confidence** label rather than an **I don't know** abstention. This gave the user a partially relevant fact rather than an explicit notice that 2024 records do not exist.

### 4. Root Cause Attribution
- **Primary Cause**: **Decision Rule Precedence (Generation Extrapolation & Rule Hierarchy)**.
- **Mechanism**: The decision engine prioritized partial qualification over absolute temporal abstention when a query specifies an out-of-bounds timestamp.

### 5. Concrete Engineering Fix
- **Fix**: **Temporal Range Gating in the Decision Engine**.
- **Implementation**: If a query contains an explicit year/date anchor ($T_{\text{query}}$) and the retrieved evidence explicitly establishes that records only exist for $T_{\text{doc}} > T_{\text{query}}$, route the decision directly to `I don't know` to prevent temporal extrapolation.

---

## Summary of Diagnostic Lessons & Architectural Remediation

| Failure Mode | Pipeline Stage | Primary Flaw | Architectural Solution |
| :--- | :--- | :--- | :--- |
| **Multi-Hop Premise** | Verification | Single-passage chunk isolation in NLI | Multi-passage premise fusion |
| **Semantic Distractor** | Retrieval | Dense vector bias toward numerical matches | Hybrid RRF + Symbolic Entity Guarding |
| **Synonym Cutoff** | Threshold | Fixed static threshold ($0.82$) on enterprise jargon | Domain alias table + Conformal Calibration |
| **Temporal Extrapolation**| Decision Engine| Qualified answer instead of hard abstention | Temporal range validation gate |
