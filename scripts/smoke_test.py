from pathlib import Path

from app.agent.agent import SupportAgent
from app.knowledge.parser import load_knowledge_base
from app.knowledge.retriever import KnowledgeRetriever
from app.llm.client import LLMClient


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

    session = agent.create_session()

    response = agent.handle(
        "How long does a regular customer "
        "have to return an unused backpack?",
        session,
    )

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.answer)

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source in response.sources:
        print(
            f"- {source.filename} "
            f"| {source.heading}"
        )

    print("\n" + "=" * 60)
    print("HANDOFF")
    print("=" * 60)

    print(response.handoff)


if __name__ == "__main__":
    main()