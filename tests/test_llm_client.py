from app.agent.models import AgentResponse, SourceReference


def test_agent_response_defaults():
    response = AgentResponse(
        answer="Test answer."
    )

    assert response.answer == "Test answer."
    assert response.sources == ()
    assert response.handoff is False


def test_agent_response_supports_sources():
    response = AgentResponse(
        answer="Returns are allowed within 30 days.",
        sources=(
            SourceReference(
                filename="01-returns-policy-current.md",
                heading="Returns Policy > Standard return window",
            ),
        ),
    )

    assert len(response.sources) == 1
    assert (
        response.sources[0].filename
        == "01-returns-policy-current.md"
    )


def test_agent_response_supports_handoff():
    response = AgentResponse(
        answer="A human needs to review this.",
        handoff=True,
    )

    assert response.handoff is True