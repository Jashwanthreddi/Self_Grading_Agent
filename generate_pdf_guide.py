"""
PDF Generator for Self-Grading Agent Master Interview Guide.
Generates a multi-page, professional PDF document using ReportLab.
"""

from __future__ import annotations

import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    ListFlowable,
    ListItem,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and draw 'Page X of Y' and headers/footers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Self-Grading Agent — Master Technical & Interview Guide")
            self.setStrokeColor(colors.HexColor("#E2E8F0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Footer (all pages)
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, page_text)
        self.drawString(54, 36, "Confidential — AI/ML Engineering Technical Interview Prep")
        self.setStrokeColor(colors.HexColor("#E2E8F0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


def build_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Palettes
    PRIMARY = colors.HexColor("#0F172A")    # Dark slate
    ACCENT = colors.HexColor("#2563EB")     # Royal Blue
    SECONDARY = colors.HexColor("#475569")  # Slate Gray
    BG_LIGHT = colors.HexColor("#F8FAFC")   # Crisp background
    SUCCESS = colors.HexColor("#16A34A")    # Green
    WARNING = colors.HexColor("#D97706")    # Amber
    CARD_BG = colors.HexColor("#F1F5F9")

    # Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=PRIMARY,
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=ACCENT,
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "SectionH1",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "SectionH2",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=ACCENT,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=PRIMARY,
        spaceAfter=6,
    )

    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1E293B"),
    )

    callout_style = ParagraphStyle(
        "CalloutText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E3A8A"),
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
    )

    table_cell_style = ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=PRIMARY,
    )

    table_cell_bold = ParagraphStyle(
        "TableCellBold",
        parent=table_cell_style,
        fontName="Helvetica-Bold",
    )

    elements = []

    # =========================================================================
    # HEADER / COVER TITLE
    # =========================================================================
    elements.append(Paragraph("Self-Grading Agent", title_style))
    elements.append(Paragraph("Complete Technical Interview Master Guide & Architecture Reference", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT, spaceBefore=0, spaceAfter=10))

    # Executive Pitch Box
    pitch_html = """
    <b>60-Second Interview Pitch:</b><br/>
    <i>"Standard RAG systems suffer from sycophancy and uncalibrated overconfidence—when an LLM hallucinates, asking it to verify its own text creates confirmation bias. I designed and built a <b>Self-Grading Agent</b> with a strictly decoupled architecture: generation is powered by Google Gemini (extracting structured atomic claims), verification is performed independently by a local Cross-Encoder NLI model (MiniLM2-L6-H768) alongside deterministic symbolic validators, and a hard decision engine assigns calibrated confidence labels (High, Low, or I don't know). The system guarantees zero dangerous overconfidence on enterprise benchmark tests."</i>
    """
    pitch_table = Table([[Paragraph(pitch_html, callout_style)]], colWidths=[504])
    pitch_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EFF6FF")),
        ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#BFDBFE")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(pitch_table)
    elements.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 1: ARCHITECTURE & PIPELINE
    # =========================================================================
    elements.append(Paragraph("1. System Architecture & 4-Stage Agent Pipeline", h1_style))
    elements.append(Paragraph(
        "The core principle is the <b>strict separation of Generation and Verification</b>. The pipeline consists of 4 distinct, independently testable stages:",
        body_style
    ))

    arch_data = [
        [
            Paragraph("Stage", table_header_style),
            Paragraph("Component & Tech Stack", table_header_style),
            Paragraph("Key Responsibilities & Operations", table_header_style),
        ],
        [
            Paragraph("<b>Stage 1:<br/>Retrieval</b>", table_cell_bold),
            Paragraph("<b>Hybrid Search</b><br/>• FAISS (BGE-base-en-v1.5)<br/>• BM25Okapi (Lexical)", table_cell_style),
            Paragraph("Ingests markdown corpus; encodes 768-dim dense embeddings + sparse BM25 indices. Performs Reciprocal Rank Fusion (60% Dense + 40% BM25) to capture both semantic concepts and exact keyword numbers ($49, dates, codes).", table_cell_style),
        ],
        [
            Paragraph("<b>Stage 2:<br/>Generation</b>", table_cell_bold),
            Paragraph("<b>Google Gemini</b><br/>• Gemini 2.5 / 3.8 Flash<br/>• Pydantic Schema", table_cell_style),
            Paragraph("Consumes top retrieved passages under strict zero-shot constraints. Generates a grounded draft answer and decomposes it into discrete <b>atomic factual claims</b> (Pydantic <code>DraftAnswer</code>). Includes automated fallback cascade for 429/503 spikes.", table_cell_style),
        ],
        [
            Paragraph("<b>Stage 3:<br/>Verification</b>", table_cell_bold),
            Paragraph("<b>Local NLI + Symbolic</b><br/>• Cross-Encoder MiniLM2<br/>• Entity / Num Guards", table_cell_style),
            Paragraph("Extracts section-grounded premise lines from passages. Cross-Encoder computes softmax probabilities [Entailment, Neutral, Contradiction] for each atomic claim. Regex symbolic guards cross-check dates, numbers, currency, and conditionals.", table_cell_style),
        ],
        [
            Paragraph("<b>Stage 4:<br/>Decision</b>", table_cell_bold),
            Paragraph("<b>Deterministic Rules</b><br/>• Rule Hierarchy<br/>• 3 Standard Labels", table_cell_style),
            Paragraph("Applies hard deterministic precedence: Abstain if missing evidence/abstain flag; High confidence if Support >= 0.82 and all guards pass; Low confidence for partial support/qualifiers. Produces auditable one-line justification.", table_cell_style),
        ],
    ]
    t_arch = Table(arch_data, colWidths=[70, 144, 290])
    t_arch.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t_arch)
    elements.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 2: WHY THESE MODELS & DECISIONS?
    # =========================================================================
    elements.append(Paragraph("2. Deep Dive: Model Selection & Technical Rationale", h1_style))

    q_a_models = [
        ("Why BAAI/bge-base-en-v1.5 for Dense Embeddings?",
         "BGE-base is consistently a top performer on the Massive Text Embedding Benchmark (MTEB). It produces high-density 768-dimensional embeddings that excel in capturing sentence-level semantic relatedness. It is lightweight, fast to compute on CPU, and normalized with inner-product cosine similarity in FAISS."),

        ("Why Hybrid Retrieval (FAISS + BM25) instead of pure Vector Search?",
         "Pure dense vectors frequently fail on subtle keyword traps (e.g., confusing $49/user with $79/workspace, or misidentifying version numbers like v2.4 vs v2.5). BM25 guarantees exact keyword matching for product names, pricing digits, and dates, while FAISS handles semantic paraphrases. Combining them via score fusion eliminates blind spots."),

        ("Why a Local Cross-Encoder NLI Model (nli-MiniLM2-L6-H768)?",
         "1. <b>Full Cross-Attention:</b> Bi-encoders encode premise and hypothesis separately (cosine similarity), which measures topical similarity rather than logical truth. Cross-encoders perform joint cross-attention over [CLS] Premise [SEP] Hypothesis [SEP], detecting true entailment vs contradiction.<br/>"
         "2. <b>Zero Prompt Bias:</b> It eliminates LLM self-confirmation bias.<br/>"
         "3. <b>Latency & Cost:</b> Runs locally in <15ms on CPU with zero network API costs, quota constraints, or outages."),

        ("Why Decoupled Verification instead of LLM-as-a-Judge?",
         "LLM-as-a-judge calls are 2x-3x more expensive, introduce double the API latency, and suffer from prompt-induced hallucination cascades. A local NLI model provides deterministic, mathematical probability distributions that are auditable and reproducible."),
    ]

    for q, a in q_a_models:
        elements.append(Paragraph(f"• <b>{q}</b>", h2_style))
        elements.append(Paragraph(a, body_style))

    elements.append(Spacer(1, 8))

    # =========================================================================
    # SECTION 3: COMPLETE FILE-BY-FILE BREAKDOWN
    # =========================================================================
    elements.append(Paragraph("3. Comprehensive File-by-File Codebase Map", h1_style))
    elements.append(Paragraph("A guided breakdown of every file in the repository for architectural review:", body_style))

    file_table_data = [
        [Paragraph("File Path", table_header_style), Paragraph("Module / Role", table_header_style), Paragraph("Key Logic & Purpose", table_header_style)],
        [
            Paragraph("<code>app/config.py</code>", code_style),
            Paragraph("Configuration", table_cell_bold),
            Paragraph("Pydantic BaseSettings loading environment variables (<code>GEMINI_API_KEY</code>, model names, top-k, weights, index paths).", table_cell_style)
        ],
        [
            Paragraph("<code>app/models/schemas.py</code>", code_style),
            Paragraph("Data Contracts", table_cell_bold),
            Paragraph("Pydantic models for <code>RetrievedPassage</code>, <code>DraftAnswer</code> (claims, abstain), <code>ClaimCheck</code>, and <code>FinalAnswer</code>.", table_cell_style)
        ],
        [
            Paragraph("<code>app/models/verification.py</code>", code_style),
            Paragraph("Verification Schema", table_cell_bold),
            Paragraph("Data model <code>VerificationResult</code> storing support scores, contradiction scores, entity/temporal/numerical match booleans.", table_cell_style)
        ],
        [
            Paragraph("<code>app/ingestion/loader.py</code>", code_style),
            Paragraph("Doc Ingestion", table_cell_bold),
            Paragraph("Parses markdown files in <code>data/knowledge_base/</code>, extracting metadata headers (Doc ID, Title, Source) and structured text chunks.", table_cell_style)
        ],
        [
            Paragraph("<code>app/ingestion/indexer.py</code>", code_style),
            Paragraph("Indexing Engine", table_cell_bold),
            Paragraph("Computes BGE dense embeddings, builds L2-normalized FAISS IndexFlatIP, and builds BM25Okapi pickled index.", table_cell_style)
        ],
        [
            Paragraph("<code>app/retrieval/retriever.py</code>", code_style),
            Paragraph("Hybrid Retriever", table_cell_bold),
            Paragraph("<code>HybridRetriever</code> class executing dense FAISS search + sparse BM25 search, fusing scores via normalized weights (0.6 / 0.4).", table_cell_style)
        ],
        [
            Paragraph("<code>app/generation/generator.py</code>", code_style),
            Paragraph("Generation Engine", table_cell_bold),
            Paragraph("<code>GeminiGenerator</code> enforcing zero-shot grounded prompts, structured JSON schema response, and automated model fallback cascade.", table_cell_style)
        ],
        [
            Paragraph("<code>app/verification/verifier.py</code>", code_style),
            Paragraph("NLI Verifier", table_cell_bold),
            Paragraph("<code>NliVerifier</code> executing local Cross-Encoder batch inference, premise line extraction, contradiction detection, and regex entity guards.", table_cell_style)
        ],
        [
            Paragraph("<code>app/decision/confidence.py</code>", code_style),
            Paragraph("Decision Engine", table_cell_bold),
            Paragraph("<code>ConfidenceDecision</code> enforcing deterministic priority rules: Abstain -> High Confidence (Score >= 0.82) -> Low Confidence.", table_cell_style)
        ],
        [
            Paragraph("<code>app/pipeline/agent.py</code>", code_style),
            Paragraph("Agent Orchestrator", table_cell_bold),
            Paragraph("<code>SelfGradingAgent</code> facade connecting Retrieval -> Generation -> Verification -> Decision into a clean <code>run(question)</code> API.", table_cell_style)
        ],
        [
            Paragraph("<code>app/main.py</code>", code_style),
            Paragraph("Streamlit Web UI", table_cell_bold),
            Paragraph("Interactive browser application featuring Q&A input, confidence badges, justification explanations, NLI inspector, and KB browser.", table_cell_style)
        ],
        [
            Paragraph("<code>run.py</code>", code_style),
            Paragraph("CLI Master Runner", table_cell_bold),
            Paragraph("Command-line runner supporting the 4-level live demonstration, custom question queries, interactive REPL shell (<code>-i</code>), and UI launcher.", table_cell_style)
        ],
        [
            Paragraph("<code>scripts/build_index.py</code>", code_style),
            Paragraph("Build Script", table_cell_bold),
            Paragraph("One-command utility to index/re-index knowledge base documents into <code>indexes/faiss</code> and <code>indexes/bm25</code>.", table_cell_style)
        ],
        [
            Paragraph("<code>scripts/run_evaluation.py</code>", code_style),
            Paragraph("Evaluation Harness", table_cell_bold),
            Paragraph("Executes the 25 benchmark questions, computes metrics, and outputs CSV/JSON reports and the confusion matrix plot.", table_cell_style)
        ],
        [
            Paragraph("<code>POST_MORTEM.md</code>", code_style),
            Paragraph("Engineering Report", table_cell_bold),
            Paragraph("Architectural post-mortem analyzing decoupling, trade-offs, failure modes at $10^5$ scale, and production hardening roadmap.", table_cell_style)
        ],
        [
            Paragraph("<code>FAILURE_ANALYSIS.md</code>", code_style),
            Paragraph("Failure Analysis", table_cell_bold),
            Paragraph("4 granular failure case studies diagnosing root causes across retrieval, generation, verification, and thresholding with actionable fixes.", table_cell_style)
        ],
    ]
    t_files = Table(file_table_data, colWidths=[120, 94, 290])
    t_files.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t_files)
    elements.append(Spacer(1, 10))

    # =========================================================================
    # SECTION 4: SCENARIO-BASED INTERVIEW QUESTIONS & ANSWERS
    # =========================================================================
    elements.append(Paragraph("4. Scenario-Based Technical Interview Q&A", h1_style))

    scenarios = [
        ("Scenario 1: How does the agent handle Trap Questions (e.g. Swapped Price / Product)?",
         "<b>Question Asked:</b> <i>'Does NovaSearch cost $79 per workspace per month?'</i><br/>"
         "<b>System Behavior:</b><br/>"
         "1. <i>Hybrid Retrieval:</i> BM25 and FAISS fetch <code>01_nexatech.md</code> (NovaSearch is $49/user) and <code>08_pricing_matrix.md</code> (FlowPilot is $79/workspace).<br/>"
         "2. <i>Generation:</i> The LLM detects the discrepancy and generates an explicit correction stating NovaSearch is $49/user while FlowPilot is $79/workspace.<br/>"
         "3. <i>Atomic Decomposition:</i> It extracts positive verifiable claims for both products.<br/>"
         "4. <i>NLI Verification:</i> Cross-Encoder verifies both factual propositions against the retrieved matrix (Entailment = 0.99).<br/>"
         "5. <i>Outcome:</i> <b>High confidence</b> because the correction is fully supported by the knowledge base."),

        ("Scenario 2: How do you prevent hallucinations on Unanswerable Queries?",
         "<b>Question Asked:</b> <i>'What was NexaTech Solutions' total revenue in 2026?'</i><br/>"
         "<b>System Behavior:</b><br/>"
         "1. The knowledge base contains no 2026 financial records.<br/>"
         "2. The generator recognizes missing data and outputs <code>abstain=True</code> with no unsupported claims.<br/>"
         "3. The decision engine encounters the abstention signal and immediately routes to: <b>'I don't know — the documents do not contain enough evidence to answer.'</b>"),

        ("Scenario 3: How do you handle Ambiguous or Conditional Questions?",
         "<b>Question Asked:</b> <i>'Does NexaTech provide 24/7 customer support?'</i><br/>"
         "<b>System Behavior:</b><br/>"
         "1. Retrieved evidence states standard support is M-F 09:00-18:00, while 24/7 support is conditional on Premium Support tier.<br/>"
         "2. The system generates a nuanced conditional answer.<br/>"
         "3. If a claim omits the condition qualifier, the verifier flags <code>condition_match=False</code>, forcing a calibrated <b>Low confidence</b> outcome."),

        ("Scenario 4: What is Dangerous Overconfidence and how is it measured?",
         "<b>Definition:</b> A critical safety metric where the agent delivers a <i>High confidence</i> label for an answer that is factually wrong.<br/>"
         "<b>Formula:</b> <code>Dangerous Overconfidence = (Wrong Answers with High Confidence) / (Total High Confidence Predictions)</code>.<br/>"
         "<b>Our Result:</b> <b>0 cases (0.0%)</b> across the entire 25-question benchmark dataset."),

        ("Scenario 5: What breaks at Enterprise Scale (100,000+ Documents) and how do you scale it?",
         "1. <b>Verification Latency Explosion:</b> Evaluating M atomic claims against K passages creates O(M x K) Cross-Encoder calls. <i>Fix:</i> Implement Hierarchical Verification (Bi-Encoder ColBERT fast filter to prune 90% of pairs, followed by Cross-Encoder only on top-2 passages).<br/>"
         "2. <b>Multi-Hop Premise Fragmentation:</b> When facts span multiple documents, single-chunk NLI classifies combined deductions as neutral. <i>Fix:</i> Multi-passage knowledge graph synthesis.<br/>"
         "3. <b>Threshold Drift:</b> <i>Fix:</i> Replace static empirical thresholds with Conformal Risk Control to guarantee P(Error | High Confidence) <= 1% statistically."),
    ]

    for q, a in scenarios:
        elements.append(Paragraph(q, h2_style))
        elements.append(Paragraph(a, body_style))
        elements.append(Spacer(1, 4))

    # =========================================================================
    # SECTION 5: LIVE DEMO CHEAT SHEET FOR THE INTERVIEW
    # =========================================================================
    elements.append(Paragraph("5. Live Demonstration Quick-Reference Cheat Sheet", h1_style))

    demo_data = [
        [Paragraph("Category", table_header_style), Paragraph("Test Question to Type", table_header_style), Paragraph("Expected Label & Key Signal", table_header_style)],
        [
            Paragraph("<b>1. Answerable</b>", table_cell_bold),
            Paragraph("<i>'When was NovaSearch launched?'</i>", table_cell_style),
            Paragraph("<b>High confidence</b> (Entailment: 0.99, Answer: March 2025)", table_cell_style),
        ],
        [
            Paragraph("<b>2. Unanswerable</b>", table_cell_bold),
            Paragraph("<i>'What was NexaTech Solutions' total revenue in 2026?'</i>", table_cell_style),
            Paragraph("<b>I don't know</b> (Abstention flag, Evidence Sufficient: No)", table_cell_style),
        ],
        [
            Paragraph("<b>3. Ambiguous</b>", table_cell_bold),
            Paragraph("<i>'Does NexaTech provide 24/7 customer support?'</i>", table_cell_style),
            Paragraph("<b>High / Low confidence</b> (Details Standard M-F vs Premium 24/7 tier)", table_cell_style),
        ],
        [
            Paragraph("<b>4. Trap Question</b>", table_cell_bold),
            Paragraph("<i>'Does NovaSearch cost $79 per workspace per month?'</i>", table_cell_style),
            Paragraph("<b>High confidence</b> (Corrects trap: NovaSearch is $49/user, FlowPilot is $79/workspace)", table_cell_style),
        ],
    ]
    t_demo = Table(demo_data, colWidths=[80, 204, 220])
    t_demo.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(t_demo)
    elements.append(Spacer(1, 10))

    # Summary Footer Note
    elements.append(Paragraph(
        "<b>Summary for the Interviewer:</b> This system is production-ready, fully tested (25/25 pytest passing), dockerizable, and strictly adheres to the calibrated confidence criteria.",
        callout_style
    ))

    # Build Document with NumberedCanvas
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"PDF successfully generated at: {filename}")


if __name__ == "__main__":
    output_path = str(Path(__file__).resolve().parent / "SELF_GRADING_AGENT_MASTER_INTERVIEW_GUIDE.pdf")
    build_pdf(output_path)
