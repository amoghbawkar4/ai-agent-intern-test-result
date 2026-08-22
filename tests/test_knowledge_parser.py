from pathlib import Path

import pytest

from app.knowledge.parser import (
    _parse_front_matter,
    _split_into_sections,
    load_knowledge_base,
    parse_document,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = REPO_ROOT / "knowledge-base"


def test_parse_front_matter():
    text = """---
document_id: TEST-1
title: Test Policy
status: active
---

# Test

Content.
"""

    metadata, content = _parse_front_matter(text)

    assert metadata["document_id"] == "TEST-1"
    assert metadata["title"] == "Test Policy"
    assert metadata["status"] == "active"
    assert "# Test" in content


def test_parse_front_matter_without_front_matter():
    metadata, content = _parse_front_matter(
        "# Test\n\nContent."
    )

    assert metadata == {}
    assert content == "# Test\n\nContent."


def test_parse_front_matter_requires_closing_delimiter():
    with pytest.raises(ValueError):
        _parse_front_matter(
            "---\ntitle: Broken\n# Missing close"
        )


def test_sections_preserve_headings():
    content = """# Returns Policy

Introduction.

## Standard return window

Customers get 30 days.

## Item condition

Items must be unused.
"""

    sections = _split_into_sections(
        content
    )

    assert len(sections) == 3

    assert sections[0][0] == (
        "Returns Policy"
    )

    assert sections[1][0] == (
        "Returns Policy > Standard return window"
    )

    assert sections[2][0] == (
        "Returns Policy > Item condition"
    )


def test_parse_document_preserves_metadata(
    tmp_path,
):
    path = tmp_path / "policy.md"

    path.write_text(
        """---
document_id: TEST-2
title: Test Policy
status: active
audience: customer
policy_authority: official
effective_date: 2026-01-01
---

# Test Policy

## Rule

Customers may do something.
""",
        encoding="utf-8",
    )

    chunks = parse_document(path)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.document_id == "TEST-2"
    assert chunk.filename == "policy.md"
    assert chunk.title == "Test Policy"
    assert chunk.heading == (
        "Test Policy > Rule"
    )
    assert chunk.status == "active"
    assert chunk.audience == "customer"
    assert chunk.policy_authority == "official"
    assert chunk.effective_date == "2026-01-01"
    assert chunk.metadata["document_id"] == "TEST-2"


def test_knowledge_base_loads_expected_documents():
    chunks = load_knowledge_base(
        KNOWLEDGE_BASE
    )

    assert len(chunks) == 53

    filenames = {
        chunk.filename
        for chunk in chunks
    }

    assert (
        "01-returns-policy-current.md"
        in filenames
    )

    assert (
        "09-trailplus-membership.md"
        in filenames
    )

    assert (
        "14-internal-content-migration-notes.md"
        in filenames
    )