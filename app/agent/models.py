from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceReference:
    """A customer-visible knowledge-base source."""

    filename: str
    heading: str


@dataclass(frozen=True)
class ToolCall:
    """A sanitized record of an application tool invocation."""

    name: str
    arguments: dict[str, str]


@dataclass(frozen=True)
class AgentResponse:
    """Structured response returned by the support agent."""

    answer: str
    sources: tuple[SourceReference, ...] = ()
    handoff: bool = False
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass
class ConversationState:
    """
    Session-local conversation state.

    Only relevant context is retained here. Sessions are isolated.
    """

    messages: list[dict[str, str]] = field(
        default_factory=list
    )

    last_order_id: str | None = None

    last_topic: str | None = None

    tool_calls: list[ToolCall] = field(
        default_factory=list
    )