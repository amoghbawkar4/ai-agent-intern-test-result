# Aster & Row - Reliable RAG Support Agent

A reliable AI customer-support agent built for the Aster & Row intern take-home assignment.

The agent uses Retrieval-Augmented Generation (RAG), deterministic safety rules, conversation context, and an order lookup tool to answer customer questions using the supplied knowledge base and mock order data.

## Features

- Retrieval-Augmented Generation (RAG)
- Current vs legacy policy handling
- TrailPlus customer-segment handling
- Order lookup with privacy protection
- Conversation/session context
- Prompt-injection protection
- Source-conflict detection
- Human handoff for insufficient or conflicting information
- Deterministic safety and routing rules

## Tech Stack

- Python
- OpenAI API
- Retrieval-based knowledge system
- Markdown knowledge base
- JSON mock order database
- pytest

## Setup

Create a virtual environment:

    python -m venv .venv
    .venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

Create a `.env` file in the project root:

    OPENAI_API_KEY=your_api_key_here
    OPENAI_MODEL=gpt-5-mini

Do not commit `.env` or your API key.

## Run

Run the smoke test:

    python -m scripts.smoke_test

Run the unit tests:

    pytest -q

Run the evaluation using the real LLM:

    python -m scripts.evaluate --real

## Results

### Unit Tests

    52 passed

### Evaluation

    TOTAL: 15/15
    All evaluation cases passed.

The evaluation was completed using the real OpenAI LLM.

## Reliability

The agent is designed to:

- Answer using retrieved company evidence
- Avoid inventing order information
- Protect internal customer and order data
- Treat retrieved prompt-injection content as untrusted
- Detect genuine source conflicts
- Avoid unsupported claims when information is insufficient
- Escalate cases requiring human review
- Preserve relevant conversation context

## Known Limitations

- Uses the supplied local knowledge base.
- Uses mock order data instead of a real ecommerce backend.
- Does not perform real refunds, cancellations, or replacements.
- Human handoff is represented by an application flag.
- Conversation state is session-based.
- A valid OpenAI API key is required.

## Demo

A short walkthrough demonstrating:
- Knowledge-base question with citations
- Order lookup
- Multi-turn conversation
- Safe abstention and human handoff
- Full evaluation suite
[Watch the demo video](./project%20demo.mp4)