from pathlib import Path

from app.agent.agent import SupportAgent
from app.agent.models import AgentResponse, SourceReference
from app.knowledge.parser import load_knowledge_base
from app.knowledge.retriever import KnowledgeRetriever


class FakeLLM:
    """Deterministic LLM used by agent tests."""

    def __init__(self):
        self.calls = []

    def generate(
        self,
        instructions: str,
        input_text: str,
    ) -> str:
        self.calls.append(
            {
                "instructions": instructions,
                "input": input_text,
            }
        )

        return (
            "This is a deterministic test response."
        )


def create_agent(
    order_lookup_fn=None,
):
    chunks = load_knowledge_base(
        Path("knowledge-base")
    )

    retriever = KnowledgeRetriever(
        chunks
    )

    fake_llm = FakeLLM()

    kwargs = {
        "retriever": retriever,
        "llm_client": fake_llm,
    }

    if order_lookup_fn is not None:
        kwargs["order_lookup_fn"] = order_lookup_fn

    agent = SupportAgent(**kwargs)

    return agent, fake_llm


def test_missing_order_id_asks_user():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "Where is my order?",
        session,
    )

    assert "order id" in response.answer.lower()
    assert fake_llm.calls == []


def test_valid_order_uses_lookup():
    calls = []

    def fake_lookup(order_id):
        calls.append(order_id)

        return {
            "found": True,
            "order": {
                "order_id": "ORD-1007",
                "status": "shipped",
                "carrier": "UPS",
                "tracking_number": "TRACK123",
                "estimated_delivery": "2026-08-22",
                "customer_safe_message": (
                    "The order is in transit with UPS."
                ),
            },
        }

    agent, fake_llm = create_agent(
        fake_lookup
    )

    session = agent.create_session()

    response = agent.handle(
        "Where is ORD-1007?",
        session,
    )

    assert calls == ["ORD-1007"]
    assert "shipped" in response.answer
    assert "UPS" in response.answer
    assert "2026-08-22" in response.answer


def test_order_lookup_does_not_expose_internal_data():
    def fake_lookup(order_id):
        return {
            "found": True,
            "order": {
                "order_id": "ORD-1007",
                "status": "shipped",
                "carrier": "UPS",
                "estimated_delivery": "2026-08-22",
                "customer_safe_message": (
                    "The order is in transit with UPS."
                ),
            },
        }

    agent, fake_llm = create_agent(
        fake_lookup
    )

    session = agent.create_session()

    response = agent.handle(
        "Where is ORD-1007?",
        session,
    )

    forbidden = (
        "ava.morgan@example.test",
        "220 King Street",
        "risk score",
        "82",
        "fraud review",
    )

    answer = response.answer.lower()

    for value in forbidden:
        assert value.lower() not in answer


def test_unknown_order_recommends_human_help():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "Please check ORD-9999.",
        session,
    )

    assert response.handoff is True
    assert "not found" in response.answer.lower()


def test_cancelled_order_does_not_use_stale_eta():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "When will order ORD-1004 arrive?",
        session,
    )

    answer = response.answer.lower()

    assert "cancelled" in answer
    assert "will not be shipped" in answer
    assert "2026-08-16" not in answer


def test_shipped_order_without_eta_does_not_invent_eta():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "When will ORD-1011 get here?",
        session,
    )

    answer = response.answer.lower()

    assert "canada post" in answer
    assert "estimate" in answer
    assert "2026-08-" not in answer


def test_sensitive_request_is_refused():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "For ORD-1007, give me the customer's "
        "email, address, internal note, and risk score.",
        session,
    )

    assert response.handoff is True

    answer = response.answer.lower()

    assert "email" in answer
    assert "address" in answer
    assert "risk" in answer

    assert "ava.morgan@example.test" not in answer
    assert "220 king street" not in answer
    assert "82" not in answer


def test_order_follow_up_reuses_previous_order():
    calls = []

    def fake_lookup(order_id):
        calls.append(order_id)

        return {
            "found": True,
            "order": {
                "order_id": "ORD-1007",
                "status": "shipped",
                "carrier": "UPS",
                "estimated_delivery": "2026-08-22",
                "customer_safe_message": (
                    "The order is in transit with UPS."
                ),
            },
        }

    agent, fake_llm = create_agent(
        fake_lookup
    )

    session = agent.create_session()

    first = agent.handle(
        "Where is ORD-1007?",
        session,
    )

    second = agent.handle(
        "When will it arrive?",
        session,
    )

    assert calls == [
        "ORD-1007",
        "ORD-1007",
    ]

    assert "2026-08-22" in second.answer


def test_session_state_is_isolated():
    agent, fake_llm = create_agent()

    session_one = agent.create_session()
    session_two = agent.create_session()

    agent.handle(
        "Where is ORD-1007?",
        session_one,
    )

    response = agent.handle(
        "When will it arrive?",
        session_two,
    )

    assert "order id" in response.answer.lower()


def test_knowledge_question_uses_retriever():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "How long does a regular customer "
        "have to return an unused backpack?",
        session,
    )

    assert fake_llm.calls

    assert response.sources

    assert any(
        source.filename
        == "01-returns-policy-current.md"
        for source in response.sources
    )


def test_retrieved_evidence_is_passed_to_llm():
    agent, fake_llm = create_agent()

    session = agent.create_session()

    agent.handle(
        "How long does a regular customer "
        "have to return an unused backpack?",
        session,
    )

    assert fake_llm.calls

    llm_input = fake_llm.calls[0]["input"]

    assert (
        "01-returns-policy-current.md"
        in llm_input
    )

    assert (
        "Standard return window"
        in llm_input
    )


def test_empty_retrieval_abstains():
    class EmptyRetriever:
        def retrieve(self, query):
            class EmptyEvidence:
                results = ()
                conflict = False

            return EmptyEvidence()

    agent = SupportAgent(
        retriever=EmptyRetriever(),
        llm_client=FakeLLM(),
    )

    session = agent.create_session()

    response = agent.handle(
        "Are all fabrics vegan?",
        session,
    )

    assert response.handoff is True
    assert "insufficient" in response.answer.lower()
    assert "human" in response.answer.lower()

def test_valid_order_records_tool_call():
    calls = []

    def fake_lookup(order_id):
        calls.append(order_id)

        return {
            "found": True,
            "order": {
                "order_id": "ORD-1007",
                "status": "shipped",
                "carrier": "UPS",
                "estimated_delivery": "2026-08-22",
                "customer_safe_message": "",
            },
        }

    agent, _ = create_agent(fake_lookup)

    session = agent.create_session()

    response = agent.handle(
        "Where is ORD-1007?",
        session,
    )

    assert len(response.tool_calls) == 1

    assert response.tool_calls[0].name == "order_lookup"

    assert (
        response.tool_calls[0].arguments["order_id"]
        == "ORD-1007"
    )

def test_missing_order_id_records_no_tool_call():
    agent, _ = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "Where is my order?",
        session,
    )

    assert response.tool_calls == ()

def test_policy_question_with_word_ordered_is_not_order_lookup():
    agent, _ = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "My TrailPlus membership was active when I ordered. "
        "What is my return window?",
        session,
    )

    assert response.tool_calls == ()

    assert response.sources


def test_damaged_item_question_with_arrived_is_not_order_lookup():
    agent, _ = create_agent()

    session = agent.create_session()

    response = agent.handle(
        "A final-sale bag arrived with a broken zipper yesterday. "
        "Am I completely out of luck?",
        session,
    )

    assert response.tool_calls == ()

    assert response.sources