"""
Self-Grading Agent - Master Runner
==================================
A clean, simple entry point for running and demonstrating the Self-Grading Agent.

Usage:
  python run.py                  # Runs 4 demo questions covering all categories
  python run.py "Your question"  # Answers a specific question
  python run.py -i               # Interactive question-answering mode
  python run.py --ui             # Launches the Streamlit Web Application
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Fix Windows console encoding for clean display
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.agent import SelfGradingAgent

# The 4 benchmark question types required for demonstration
DEMO_QUESTIONS = [
    {
        "level": "Level 1: Answerable Question (Full Evidence Available)",
        "question": "When was NovaSearch launched?",
        "expected": "High confidence — the answer is clearly supported by the sources.",
        "note": "Evidence clearly states NovaSearch launched in March 2025.",
    },
    {
        "level": "Level 2: Unanswerable Question (Evidence Absent)",
        "question": "What was NexaTech Solutions' total revenue in 2026?",
        "expected": "I don't know — the documents do not contain enough evidence to answer.",
        "note": "The knowledge base has no 2026 financial records; the agent must abstain.",
    },
    {
        "level": "Level 3: Partially Supported / Ambiguous / Conditional Question",
        "question": "Does NexaTech provide 24/7 customer support?",
        "expected": "Low confidence (or conditional explanation)",
        "note": "24/7 support is conditional on Premium Support tier (standard is M-F 09:00-18:00).",
    },
    {
        "level": "Level 4: Trap Question (Contains False Premise)",
        "question": "Does NovaSearch cost $79 per workspace per month?",
        "expected": "High confidence (Corrects trap: NovaSearch is $49/user, FlowPilot is $79/workspace)",
        "note": "The question contains an incorrect price/product swap; the agent corrects it.",
    },
]


def display_result(result: dict) -> None:
    """Print the question, answer, confidence, and reasoning clearly."""
    question = result.get("question", "")
    answer = result.get("answer", "")
    confidence = result.get("confidence", "")
    reason = result.get("reason", "")
    verification = result.get("verification")

    print("\n" + "=" * 76)
    print(f"QUESTION: {question}")
    print("=" * 76)

    # 1. Final Confidence Decision
    print("\n[CONFIDENCE LABEL]")
    print(f"  --> {confidence}")

    # 2. Generated Answer
    print("\n[FINAL ANSWER]")
    print(f"  {answer}")

    # 3. Justification Reason
    print("\n[REASON / JUSTIFICATION]")
    print(f"  {reason}")

    # 4. Independent Verification Breakdown
    if verification:
        v_dict = (
            verification.model_dump()
            if hasattr(verification, "model_dump")
            else (verification if isinstance(verification, dict) else {})
        )
        print("\n[INDEPENDENT VERIFICATION SIGNALS]")
        print(f"  - Support Score:       {v_dict.get('support_score', 0.0):.2f}")
        print(f"  - Contradiction Score: {v_dict.get('contradiction_score', 0.0):.2f}")
        print(f"  - Evidence Sufficient: {'Yes' if v_dict.get('evidence_sufficient') else 'No'}")

        claims = v_dict.get("claim_results", [])
        if claims:
            print("\n  Atomic Claim Entailment:")
            for idx, claim in enumerate(claims, start=1):
                c_dict = claim if isinstance(claim, dict) else (claim.model_dump() if hasattr(claim, "model_dump") else {})
                c_text = c_dict.get("claim", "")
                c_verdict = c_dict.get("verdict", "").upper()
                c_ent = c_dict.get("entailment", 0.0)
                print(f"    [{c_verdict}] Claim {idx}: \"{c_text}\" (Entailment: {c_ent:.2f})")

    print("=" * 76 + "\n")


def run_demo(agent: SelfGradingAgent) -> None:
    """Run through the 4 question levels sequentially."""
    print("\n" + "#" * 76)
    print("  SELF-GRADING AGENT - 4-LEVEL LIVE DEMONSTRATION")
    print("#" * 76)

    for item in DEMO_QUESTIONS:
        print(f"\n>>> Running: {item['level']}")
        print(f">>> Expected: {item['expected']}")
        print(f">>> Note:     {item['note']}")

        result = agent.run(item["question"])
        display_result(result)


def run_interactive(agent: SelfGradingAgent) -> None:
    """Interactive loop for asking arbitrary questions."""
    print("\n" + "=" * 76)
    print("  Self-Grading Agent - Interactive Mode")
    print("  Type any question to ask against the knowledge base.")
    print("  Type 'demo' to run the 4 question levels.")
    print("  Type 'exit' to quit.")
    print("=" * 76)

    while True:
        try:
            user_input = input("\nEnter Question > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting...")
                break
            if user_input.lower() == "demo":
                run_demo(agent)
                continue

            result = agent.run(user_input)
            display_result(result)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break
        except Exception as exc:
            print(f"\n[ERROR]: {exc}")


def main() -> None:
    args = sys.argv[1:]

    if args and args[0] in ("--ui", "ui", "streamlit"):
        os.system(f'"{sys.executable}" -m streamlit run app/main.py')
        return

    print("Initializing Self-Grading Agent...")
    agent = SelfGradingAgent()
    print("Agent ready.\n")

    if not args:
        # Default behavior: run the 4-level demo so the user immediately sees results
        run_demo(agent)
    elif args[0] in ("-i", "--interactive", "interactive"):
        run_interactive(agent)
    elif args[0] in ("-d", "--demo", "demo"):
        run_demo(agent)
    else:
        # User passed a custom question string: python run.py "What is NovaSearch?"
        question = " ".join(args)
        result = agent.run(question)
        display_result(result)


if __name__ == "__main__":
    main()
