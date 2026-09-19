
# Self_Grading_Agent
Self-Grading Agent with Hybrid Retrieval, Independent NLI Verification, and Calibrated Confidence

# Self-Grading Agent: Calibrated Confidence & Answer Verification
### AI/ML Assignment: Evidence-Grounded Question Answering with Independent Verification

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![PyTest Status](https://img.shields.io/badge/pytest-25%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade Retrieval-Augmented Generation (RAG) question-answering system featuring an **independent Natural Language Inference (NLI) verification engine** and a **calibrated, conservative confidence decision layer**.

The system prevents hallucinations and dangerous overconfidence by independently auditing generated claims against retrieved source evidence before presenting answers to the user.

---

## Table of Contents
1. [Core Problem & Objectives](#1-core-problem--objectives)
2. [Exact Confidence Outcomes](#2-exact-confidence-outcomes)
3. [System Architecture & Agent Pipeline](#3-system-architecture--agent-pipeline)
4. [Knowledge Base & Benchmark Test Set](#4-knowledge-base--benchmark-test-set)
5. [Quick Start & Installation](#5-quick-start--installation)
6. [Running the Application & Demo Guide](#6-running-the-application--demo-guide)
7. [Calibration Evaluation & Benchmark Results](#7-calibration-evaluation--benchmark-results)
8. [Detailed Failure Analysis (3 Diagnoses)](#8-detailed-failure-analysis-3-diagnoses)
9. [Repository Structure](#9-repository-structure)
10. [Known Limitations & Dependencies](#10-known-limitations--dependencies)

---

## 1. Core Problem & Objectives

Standard RAG systems suffer from three critical safety and calibration flaws:
1. **Blind Overconfidence**: Large language models frequently assign high certainty to fabricated details, hallucinations, or ungrounded assumptions.
2. **Self-Confirmation Bias**: When an LLM is asked to verify its own generation within the same context, it tends to justify its own prior reasoning.
3. **Vulnerability to Entity Traps**: Superficially matching passages (e.g., similar products, shifted dates, altered pricing units) lead naive systems into answering incorrect premises.

### Solution
This project enforces a **strictly decoupled pipeline**:
- **Answer Generation** is performed by Google Gemini 3.8 Flash via the Google GenAI SDK, which produces draft text decomposed into atomic factual claims.
- **Answer Verification** is executed by a separate, local Cross-Encoder NLI model (`cross-encoder/nli-MiniLM2-L6-H768`) alongside deterministic consistency validators (entity, temporal, numerical, and condition checks).
- **Confidence Decision** evaluates the verification signals using strict deterministic priority rules to assign a calibrated confidence label or abstain with an explicit *"I don't know"*.

---

## 2. Exact Confidence Outcomes

For every query, the system outputs one of the three required confidence labels:

| Confidence Label | Operational Meaning | Decision Criteria |
| :--- | :--- | :--- |
| **High confidence** | The answer is clearly supported by the sources. | Verification support score $\ge 0.82$, zero contradictions ($< 0.30$), evidence is sufficient, and all entity, temporal, and numerical match checks pass. |
| **Low confidence** | The available evidence is incomplete, ambiguous, or requires an unsupported inference. | Evidence is present but support score $< 0.82$, or contains qualifiers, ambiguous conditions, or unsupported inferences. |
| **I don't know** | The documents do not contain enough evidence to answer. | Retrieved evidence is insufficient, or the generation/verification stage detects no grounded support. The system explicitly abstains. |

---

## 3. System Architecture & Agent Pipeline

```text
                                 +-------------------------+
                                 |      User Question      |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 |    Hybrid Retrieval     |
                                 |  - FAISS (BGE Embeds)   |
                                 |  - BM25Okapi (Lexical)  |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 |   Retrieved Evidence    |
                                 |   (Top-4 Grounded Chunks|
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 |    Draft Generator      |
                                 |    (Gemini 3.8 Flash)   |
                                 |  - Draft Answer         |
                                 |  - Atomic Factual Claims|
                                 |  - Abstention Flag      |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 | Independent Verification|
                                 |  - CrossEncoder NLI     |
                                 |  - Entity/Date/Num/Cond |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 |   Confidence Engine     |
                                 |   Deterministic Rules   |
                                 +------------+------------+
                                              |
                                              v
          +-----------------------------------------------------------------------+
          | FINAL OUTPUT:                                                         |
          |  • Answer Text (or Abstention)                                        |
          |  • Confidence Label (High / Low / I don't know)                       |
          |  • One-Line Justification Reason                                      |
          |  • Supporting Evidence Passages (IDs, Scores, Text)                   |
          |  • Granular Claim-by-Claim Verification Breakdown                     |
          +-----------------------------------------------------------------------+
```

### Stage-by-Stage Breakdown

1. **Hybrid Retrieval (`app/retrieval/retriever.py`)**:
   - Queries are processed concurrently through dense FAISS vector search (`BAAI/bge-base-en-v1.5`, dimension 768) and sparse BM25 (`rank-bm25`).
   - Normalized score fusion ($0.6 \times \text{Dense} + 0.4 \times \text{BM25}$) ensures high recall for semantic queries while capturing exact numerical and entity matches.
2. **Structured Answer Generation (`app/generation/generator.py`)**:
   - The LLM receives retrieved evidence chunks and produces a Pydantic `DraftAnswer` containing the answer string, a list of atomic claims, and an abstention flag.
   - Temperature is constrained to 0.1 to minimize creativity and strictly ground outputs.
3. **Independent NLI & Consistency Verification (`app/verification/verifier.py`)**:
   - Every atomic claim is evaluated against all candidate passages using `cross-encoder/nli-MiniLM2-L6-H768`, computing softmax probabilities across `[entailment, neutral, contradiction]`.
   - Deterministic consistency checks verify that names, dates, numerical values, and condition qualifiers from the premise are preserved.
4. **Deterministic Decision Engine (`app/decision/confidence.py`)**:
   - A deterministic rule hierarchy evaluates evidence sufficiency, contradiction flags, entity/temporal/numerical alignment, and threshold checks to determine the final label.

---

## 4. Knowledge Base & Benchmark Test Set

### Knowledge Base (12 Curated Documents)
The knowledge base (`data/knowledge_base/`) contains 12 markdown documents detailing 6 tech companies, product lines, pricing, releases, and operational policies:
- `01_nexatech.md` — NexaTech Solutions (NovaSearch, enterprise search, pricing, locations).
- `02_orbitanalytics.md` — Orbit Analytics (OrbitBI, ForecastPro, leadership, pricing).
- `03_zenithrobotics.md` — Zenith Robotics (WarehousePilot, control plane, hosted regions).
- `04_clearpathhealth.md` — ClearPath Health Systems (MedFlow, HIPAA compliance, CEO).
- `05_evergreenenergy.md` — Evergreen Energy Systems (SolarWatch, grid management).
- `06_helixfintech.md` — Helix FinTech (RiskLens, SLA, banking APIs).
- `07_product_catalog.md` — Master cross-company product catalog.
- `08_pricing_matrix.md` — Detailed per-seat, per-workspace, and quote pricing tiers.
- `09_release_history.md` — Chronological launch dates and feature updates (2021–2025).
- `10_security_infrastructure.md` — Encryption standards (AES-256), data center locations.
- `11_support_policies.md` — Support tiers (Standard vs Premium 24/7 critical response).
- `12_company_comparison_facts.md` — Multi-company comparative matrices.

### Benchmark Test Set (25 Questions)
The test suite (`data/evaluation/test_questions.json`) meets and exceeds all assignment minimums:

| Question Category | Count | Min Required | Description & Expected Behavior |
| :--- | :---: | :---: | :--- |
| **Answerable** | 10 | 10 | Explicitly supported in knowledge base. System must provide correct answer and assign **High confidence**. |
| **Unanswerable** | 5 | 5 | Information is absent (e.g. 2026 revenue, employee counts, codebase language). System must abstain with **"I don't know"**. |
| **Partially Supported / Ambiguous** | 5 | 5 | Conditional support or ambiguous scope (e.g., 24/7 support only for Premium). System must provide cautious answer and assign **Low confidence**. |
| **Trap Questions** | 5 | 3 | Altered entity names (*NovaAnalytics*), swapped product dates (OrbitBI vs ForecastPro), swapped prices ($49 vs $79), or altered hosting regions. System must spot trap and correct or abstain. |
| **Total** | **25** | **20–30** | Comprehensive calibration benchmark. |

---

## 5. Quick Start & Installation

### Prerequisites
- Python 3.10, 3.11, 3.12, 3.13, or 3.14
- Google Gemini API Key ([Google AI Studio](https://aistudio.google.com/))

### 1. Clone & Create Virtual Environment
```bash
git clone https://github.com/your-username/self-grading-agent.git
cd self-grading-agent

# Create virtual environment
python -m venv .venv

# Activate on Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate on Linux / macOS
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Gemini API key:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GENERATION_MODEL=gemini-3.8-flash
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
NLI_MODEL=cross-encoder/nli-MiniLM2-L6-H768
```

### 4. Build / Refresh Search Indexes
Build the FAISS vector index and BM25 index from the knowledge base:
```bash
python scripts/build_index.py
```

---

## 6. Running the Application & Demo Guide

### 6.1 Start the Web Application
Launch the interactive Streamlit user interface:
```bash
streamlit run app/main.py
```
*(Or alternatively: `streamlit run app/ui/streamlit_app.py`)*

The UI displays:
- **Interactive Question Input Area**
- **Final Grounded Answer** (or clear Abstention banner)
- **Calibrated Confidence Badge** (Green for High, Yellow for Low, Blue for Abstain)
- **Decision Justification Reason**
- **Retrieved Evidence Passages** (with dense, BM25, and hybrid scores)
- **Claim-by-Claim Verification Inspector** (Entailment, contradiction, neutral scores)

---

### 6.2 Live Demonstration Walkthrough (Reviewer Script)

During review, test the system across all 4 question categories:

#### Test 1: Answerable Question (Expected: High confidence)
- **Input Question**: `"When was NovaSearch launched?"`
- **Expected Answer**: `"NovaSearch was launched in March 2025."`
- **Expected Confidence**: `High confidence — the answer is clearly supported by the sources.`
- **Pipeline Observation**: Hybrid retriever fetches `KB-009` and `KB-001`. The NLI verifier scores claim entailment at $>0.98$. Decision engine confirms all entity and temporal checks pass.

#### Test 2: Unanswerable Question (Expected: Abstention / "I don't know")
- **Input Question**: `"What was NexaTech Solutions' total revenue in 2026?"`
- **Expected Answer**: `I don't know — the documents do not contain enough evidence to answer.`
- **Expected Confidence**: `I don't know — the documents do not contain enough evidence to answer.`
- **Pipeline Observation**: The generator recognizes the missing figure and flags `abstain=True`. The decision engine forces clean abstention.

#### Test 3: Ambiguous / Partially Supported Question (Expected: Low confidence)
- **Input Question**: `"Does NexaTech provide 24/7 customer support?"`
- **Expected Answer**: Cautious explanation noting 24/7 is available only for Premium Support critical incidents.
- **Expected Confidence**: `Low confidence — the available evidence is incomplete, ambiguous, or requires an unsupported inference.`
- **Pipeline Observation**: Verifier recognizes conditional qualifier mismatch (`condition_match=False`), preventing an un-nuanced High confidence rating.

#### Test 4: Trap Question (Expected: High confidence correction or Abstention)
- **Input Question**: `"Does NovaSearch cost $79 per workspace per month?"`
- **Expected Answer**: `"No. NovaSearch Enterprise is $49 per user per month; $79 per workspace per month is the listed price for FlowPilot Business."`
- **Expected Confidence**: `High confidence — the answer is clearly supported by the sources.`
- **Pipeline Observation**: Retrieval fetches both `01_nexatech.md` and `08_pricing_matrix.md`. The generator detects the price/product swap, and the verifier validates the negative proposition.

#### Test 5: Unseen Reviewer Question
- **Input Question**: `"What encryption standard is used for data in transit by ClearPath Health Systems?"`
- The system will retrieve `10_security_infrastructure.md` (TLS 1.3), generate the grounded answer, verify the claim, and assign High confidence.

---

### 6.3 Run the Evaluation Benchmark
Run the automated evaluation harness over the 25 benchmark questions:
```bash
python scripts/run_evaluation.py
```
This executes the test set, evaluates answer correctness against reference facts, generates calibration metrics, outputs CSV/JSON reports to `evaluation_results/`, and creates the confusion matrix plot `confidence_confusion_matrix.png`.

### 6.4 Run Automated Unit Tests
Run all 25 pytest test cases verifying retrieval, schema validation, NLI checks, and decision rules:
```bash
pytest -v
```

---

## 7. Calibration Evaluation & Benchmark Results

### 7.1 Key Calibration Metrics

| Metric | Result | Target / Interpretation |
| :--- | :---: | :--- |
| **Total Benchmark Questions** | **25** | Complete test coverage |
| **Dangerous Overconfidence** | **0 cases (0.0%)** | **Zero ungrounded high-confidence answers** |
| **High-Confidence Precision** | **100.0%** | When the system claims High confidence, it is always correct |
| **Over-Cautious Abstentions** | 4 cases | System prefers cautious abstention over risk |
| **Answer Accuracy** | 80.0% | Deterministic benchmark correctness |
| **Confidence Calibration Accuracy** | 84.0% | Correct confidence bucket assignment |

### 7.2 Explicit Critical Counts
- **Dangerous Overconfidence**: **0** (The application never delivered a high-confidence answer that was factually wrong).
- **Over-Cautious Abstentions**: **4** (Cases where the knowledge base contained facts, but the verifier conservatively abstained due to strict claim phrasing or entity cross-checks).

### 7.3 Confidence Confusion Matrix

```text
Expected Label \ Predicted Label       High Confidence    Low Confidence    I Don't Know
-----------------------------------------------------------------------------------------
High confidence (Answerable/Trap)            11                  0                4
Low confidence (Ambiguous/Partial)            0                  5                0
I don't know (Unanswerable/Absents)           0                  0                5
```

### 7.4 Correctness Determination Methodology
Answer correctness is evaluated independently from confidence using strict, deterministic criteria:
1. **Unanswerable Questions**: Judged correct if and only if the system predicts *"I don't know"* confidence and the answer text contains standard abstention semantics.
2. **Ambiguous / Conditional Questions**: Judged correct if the system predicts *Low confidence* and the answer captures the required qualifying conditions (e.g., *"Premium Support"*, *"priority customers"*) without contradicting evidence.
3. **Answerable & Trap Questions**: Judged correct if the answer contains all required factual anchors (`required_keywords` / `required_facts`) and contains zero `forbidden_facts` (e.g., verifying that a trap price or swapped date was not affirmed).

---

## 8. Detailed Failure Analysis (3 Diagnoses)

The assignment requires deep failure analysis for at least three cases. Below are detailed diagnoses:

### Failure Case 1: Over-Cautious Abstention on Leadership Query (Q002)
- **Question**: `"Who is the CTO of Orbit Analytics?"`
- **Application Response**: `"I don't know — the documents do not contain enough evidence to answer."`
- **Retrieved Evidence**: `KB-002-chunk-1` (*"Orbit Analytics Leadership: Arjun Shah serves as Chief Technology Officer (CTO)..."*).
- **Predicted Confidence**: `I don't know` | **Expected Outcome**: `High confidence` (Answer: Arjun Shah).
- **Diagnosis**: **Verification Stage Failure**.
  - *What Went Wrong*: The generator produced two atomic claims: (1) *"Arjun Shah is CTO"* and (2) *"Arjun Shah has served as CTO since September 2021"*. While Claim 1 had 0.99 entailment, Claim 2 introduced a specific month (*September 2021*) not explicitly present in the chunk. The NLI model flagged Claim 2 as a contradiction/neutral, driving the overall support score down to 0.002.
  - *Primary Cause*: Verification aggregation logic (a single auxiliary ungrounded clause tainted the primary factual answer).
  - *Actionable Fix*: Implement claim-level isolation weighting: evaluate the primary subject-verb-predicate core independently from optional temporal subordinate clauses.

---

### Failure Case 2: Over-Cautious Abstention on Regional Infrastructure (Q003)
- **Question**: `"What is the standard hosted region for Zenith Robotics?"`
- **Application Response**: `"I don't know — the documents do not contain enough evidence to answer."`
- **Retrieved Evidence**: `KB-003-chunk-2` (*"Zenith Robotics hosts its primary control-plane cluster in Frankfurt, Germany..."*) and `KB-010-chunk-9` (*"Infrastructure Hosting Matrix..."*).
- **Predicted Confidence**: `I don't know` | **Expected Outcome**: `High confidence` (Answer: Frankfurt, Germany).
- **Diagnosis**: **Retrieval & Verification Threshold Interaction**.
  - *What Went Wrong*: The generator synthesized: *"Zenith Robotics' standard hosted region for its control-plane cluster is Frankfurt, Germany."* Because the document phrasing used *"control-plane cluster"* while the question asked for *"standard hosted region"*, the NLI model assigned an entailment score of 0.74, falling just below the strict `0.82` high-confidence threshold.
  - *Primary Cause*: Static threshold stringency on synonymously phrased technical infrastructure terms.
  - *Actionable Fix*: Add adaptive domain entity grounding or calibrate cross-encoder thresholds with a margin of tolerance when exact geographical entities (*Frankfurt, Germany*) match 100%.

---

### Failure Case 3: Over-Cautious Abstention on Product Launch Date (Q009)
- **Question**: `"When did RiskLens launch?"`
- **Application Response**: `"I don't know — the documents do not contain enough evidence to answer."`
- **Retrieved Evidence**: `KB-006-chunk-5` and `KB-009-chunk-8` (*"RiskLens v1.0 was officially released in October 2024 by Helix FinTech..."*).
- **Predicted Confidence**: `I don't know` | **Expected Outcome**: `High confidence` (Answer: October 2024).
- **Diagnosis**: **Generation Claim Extraction Failure**.
  - *What Went Wrong*: The generator formulated the claim as *"RiskLens launched in October 2024 as an automated underwriting tool"*. The secondary clause (*"as an automated underwriting tool"*) had weak textual overlap with the release history table chunk, causing the CrossEncoder to score neutral (0.47) and trigger abstention.
  - *Primary Cause*: Generation stage over-specifying atomic claims with extra adjectives not requested in the prompt.
  - *Actionable Fix*: Constrain draft claim generation to minimal canonical tuples: `(Subject: RiskLens, Relation: launched, Value: October 2024)`.

---

## 9. Repository Structure

```text
self-grading-agent/
├── .env.example                     # Safe environment variable configuration template
├── README.md                        # Master project documentation & reviewer guide
├── POST_MORTEM.md                   # Architectural post-mortem, trade-offs & scaling analysis
├── requirements.txt                 # Exact pinned dependency specifications
│
├── app/                             # Main application source code
│   ├── config.py                    # Pydantic Settings configuration & environment loader
│   ├── main.py                      # Primary Streamlit web application & visual dashboard
│   │
│   ├── models/                      # Pydantic data schemas & verification models
│   │   ├── schemas.py               # RetrievedPassage, DraftAnswer, ClaimCheck schemas
│   │   └── verification.py          # VerificationResult & signal containers
│   │
│   ├── ingestion/                   # Document parsing & chunking
│   │   ├── loader.py                # Markdown metadata extraction & loader
│   │   └── indexer.py               # FAISS dense indexer & BM25Okapi builder
│   │
│   ├── retrieval/                   # Search & candidate retrieval
│   │   └── retriever.py             # HybridRetriever (FAISS + BM25 score fusion)
│   │
│   ├── generation/                  # LLM answer synthesis
│   │   └── generator.py             # GeminiGenerator (structured claims & draft answer)
│   │
│   ├── verification/                # Independent answer auditing
│   │   └── verifier.py              # NliVerifier (CrossEncoder NLI + consistency checks)
│   │
│   ├── decision/                    # Calibrated confidence assignment
│   │   └── confidence.py            # Deterministic ConfidenceDecision engine
│   │
│   ├── evaluation/                  # Benchmark scoring & metrics
│   │   ├── evaluator.py             # Benchmark EvaluationRunner with rate-limit pacing
│   │   ├── metrics.py               # Accuracy, precision, calibration, confusion matrix
│   │   └── calibration.py           # Reliability diagrams & confusion matrix visualization
│   │
│   ├── pipeline/                    # Agent orchestration
│   │   └── agent.py                 # SelfGradingAgent end-to-end pipeline facade
│   │
│   └── ui/                          # Additional UI components
│       └── streamlit_app.py         # Dual-tab QA & Evaluation viewer
│
├── data/                            # Knowledge base and evaluation datasets
│   ├── knowledge_base/              # 12 curated corporate & product markdown documents
│   └── evaluation/                  # 25 curated benchmark questions (JSON & CSV)
│
├── evaluation_results/              # Cached evaluation outputs & calibration plots
│   ├── evaluation_results.json      # Complete per-question run outputs
│   ├── evaluation_results.csv       # Auditable spreadsheet view
│   ├── metrics.json                 # Summary metrics, distribution, calibration
│   └── confidence_confusion_matrix.png # Confusion matrix plot
│
├── scripts/                         # Command-line utility scripts
│   ├── build_index.py               # Rebuild FAISS & BM25 indexes
│   └── run_evaluation.py            # Execute complete 25-question evaluation suite
│
└── tests/                           # Automated test suite (25 tests)
    ├── conftest.py                  # Pytest fixtures & path setup
    ├── test_decision.py             # Unit tests for confidence decision rules
    ├── test_retrieval.py            # Unit tests for passage models & hybrid search
    └── test_verification.py         # Unit tests for verification signals & scores
```

---

## 10. Known Limitations & Dependencies

1. **Hosted API Rate Limits**: The answer generation step depends on Google Gemini API free-tier / paid quotas. When running large batch evaluations, the benchmark runner automatically spaces requests and implements retry backoff for 429 rate limits.
2. **Single-Passage NLI Scope**: The NLI verifier evaluates atomic claims against individual retrieved chunks. Complex multi-hop deductions that require combining disparate facts from multiple separate documents require multi-passage synthesis.
3. **Hardware Requirements**: Local models (`bge-base-en-v1.5` and `nli-MiniLM2-L6-H768`) run on CPU with $<1.5\text{ GB}$ RAM footprint. GPU acceleration is supported automatically via PyTorch if CUDA is available.
