from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class KnowledgeChunk:
    """A searchable section of a knowledge-base document."""

    document_id: str
    filename: str
    title: str
    heading: str
    content: str

    status: str | None
    audience: str | None
    policy_authority: str | None
    effective_date: str | None

    metadata: dict[str, Any]


def _normalize_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Normalize YAML metadata into stable Python values.

    PyYAML automatically converts ISO dates such as
    2026-01-01 into datetime.date objects. The application
    should expose metadata consistently as strings.
    """

    normalized = metadata.copy()

    for key in (
        "effective_date",
        "last_reviewed",
        "superseded_date",
    ):
        value = normalized.get(key)

        if value is not None:
            normalized[key] = str(value)

    return normalized


def _parse_front_matter(
    text: str,
) -> tuple[dict[str, Any], str]:
    """Split YAML front matter from Markdown content."""

    text = text.lstrip("\ufeff")

    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()

    closing_index = None

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise ValueError(
            "Document has opening front matter but no closing '---'."
        )

    front_matter_text = "\n".join(
        lines[1:closing_index]
    )

    content = "\n".join(
        lines[closing_index + 1:]
    )

    metadata = yaml.safe_load(
        front_matter_text
    ) or {}

    if not isinstance(metadata, dict):
        raise ValueError(
            "Front matter must contain a YAML mapping."
        )

    return _normalize_metadata(metadata), content


def _split_into_sections(
    content: str,
) -> list[tuple[str, str]]:
    """
    Split Markdown into sections while preserving heading hierarchy.
    """

    sections: list[tuple[str, str]] = []

    heading_stack: dict[int, str] = {}

    current_heading = ""
    current_lines: list[str] = []

    def flush_section() -> None:
        if not current_lines:
            return

        section_content = "\n".join(
            current_lines
        ).strip()

        if section_content:
            sections.append(
                (
                    current_heading,
                    section_content,
                )
            )

    for line in content.splitlines():
        stripped = line.strip()

        if stripped.startswith("#"):
            marker = stripped.split(" ", 1)[0]

            if marker and set(marker) == {"#"}:
                flush_section()

                level = len(marker)
                heading_text = stripped[level:].strip()

                heading_stack = {
                    existing_level: text
                    for existing_level, text
                    in heading_stack.items()
                    if existing_level < level
                }

                heading_stack[level] = heading_text

                current_heading = " > ".join(
                    heading_stack[level]
                    for level in sorted(heading_stack)
                )

                current_lines = []
                continue

        current_lines.append(line)

    flush_section()

    return sections


def parse_document(
    path: Path,
) -> list[KnowledgeChunk]:
    """Parse one Markdown knowledge-base document."""

    text = path.read_text(
        encoding="utf-8"
    )

    metadata, content = _parse_front_matter(
        text
    )

    sections = _split_into_sections(
        content
    )

    chunks: list[KnowledgeChunk] = []

    for heading, section_content in sections:
        chunks.append(
            KnowledgeChunk(
                document_id=str(
                    metadata.get(
                        "document_id",
                        "",
                    )
                ),
                filename=path.name,
                title=str(
                    metadata.get(
                        "title",
                        "",
                    )
                ),
                heading=heading,
                content=section_content,
                status=metadata.get("status"),
                audience=metadata.get("audience"),
                policy_authority=metadata.get(
                    "policy_authority"
                ),
                effective_date=metadata.get(
                    "effective_date"
                ),
                metadata=metadata.copy(),
            )
        )

    return chunks


def load_knowledge_base(
    directory: Path,
) -> list[KnowledgeChunk]:
    """Parse every Markdown document in the knowledge base."""

    chunks: list[KnowledgeChunk] = []

    for path in sorted(
        directory.glob("*.md")
    ):
        chunks.extend(
            parse_document(path)
        )

    return chunks