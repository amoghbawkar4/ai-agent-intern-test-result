from pathlib import Path
import subprocess
import sys

from app.agent.agent import SupportAgent
from app.knowledge.parser import load_knowledge_base
from app.knowledge.retriever import KnowledgeRetriever
from app.llm.client import LLMClient


def print_response(response):
    print("\n" + "=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(response.answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    if response.sources:
        for source in response.sources:
            print(f"- {source.filename} | {source.heading}")
    else:
        print("- None")

    print("\n" + "=" * 70)
    print("HANDOFF")
    print("=" * 70)
    print(response.handoff)

    if response.tool_calls:
        print("\n" + "=" * 70)
        print("TOOL CALLS")
        print("=" * 70)

        for tool_call in response.tool_calls:
            print(
                f"- {tool_call.name}: "
                f"{tool_call.arguments}"
            )


def main():
    knowledge_base = load_knowledge_base(
        Path("knowledge-base")
    )

    retriever = KnowledgeRetriever(
        knowledge_base
    )

    llm = LLMClient()

    agent = SupportAgent(
        retriever=retriever,
        llm_client=llm,
    )

    # =========================================================
    # 1. KNOWLEDGE-BASE QUESTION + CITATIONS
    # =========================================================

    print("\n")
    print("#" * 70)
    print("DEMO 1 — KNOWLEDGE BASE + CITATIONS")
    print("#" * 70)

    session = agent.create_session()

    response = agent.handle(
        "How long does a regular customer "
        "have to return an unused backpack?",
        session,
    )

    print_response(response)

    # =========================================================
    # 2. ORDER LOOKUP
    # =========================================================

    print("\n")
    print("#" * 70)
    print("DEMO 2 — ORDER LOOKUP")
    print("#" * 70)

    order_session = agent.create_session()

    response = agent.handle(
        "Where is ORD-1007?",
        order_session,
    )

    print_response(response)

    # =========================================================
    # 3. MULTI-TURN CONVERSATION
    # =========================================================

    print("\n")
    print("#" * 70)
    print("DEMO 3 — MULTI-TURN CONVERSATION")
    print("#" * 70)

    conversation_session = agent.create_session()

    print("\nUSER:")
    print("Do you ship internationally?")

    response = agent.handle(
        "Do you ship internationally?",
        conversation_session,
    )

    print("\nASSISTANT:")
    print(response.answer)

    print("\n" + "-" * 70)

    print("\nUSER:")
    print("What about Canada?")

    response = agent.handle(
        "What about Canada?",
        conversation_session,
    )

    print("\nASSISTANT:")
    print(response.answer)

    print("\nSOURCES:")
    for source in response.sources:
        print(
            f"- {source.filename} | {source.heading}"
        )

    # =========================================================
    # 4. INSUFFICIENT INFORMATION / HUMAN HANDOFF
    # =========================================================

    print("\n")
    print("#" * 70)
    print("DEMO 4 — SAFE ABSTENTION / HUMAN HANDOFF")
    print("#" * 70)

    handoff_session = agent.create_session()

    response = agent.handle(
        "Are the fabrics or adhesives in your bags vegan?",
        handoff_session,
    )

    print_response(response)

    # =========================================================
    # 5. FULL EVALUATION
    # =========================================================

    print("\n")
    print("#" * 70)
    print("DEMO 5 — EVALUATION SUITE")
    print("#" * 70)

    print("\nRunning:")
    print("python -m scripts.evaluate --real")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.evaluate",
            "--real",
        ],
        check=False,
    )

    print("\n" + "=" * 70)
    print(
        f"Evaluation process exited with code: "
        f"{result.returncode}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()