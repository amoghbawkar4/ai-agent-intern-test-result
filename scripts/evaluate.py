from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.agent.agent import SupportAgent
from app.agent.models import ToolCall
from app.knowledge.parser import load_knowledge_base
from app.knowledge.retriever import KnowledgeRetriever


ROOT = Path(__file__).resolve().parents[1]

VISIBLE_CASES_FILE = ROOT / "evaluation" / "visible-cases.json"
KNOWLEDGE_BASE_DIR = ROOT / "knowledge-base"

CACHE_DIR = ROOT / "evaluation" / "cache"
RESULTS_FILE = ROOT / "evaluation" / "latest-results.json"


class EvaluationLLM:
    """
    Deterministic LLM replacement for local evaluator runs.

    Normal evaluator runs must never consume API credits.
    """

    def generate(
        self,
        instructions: str,
        input_text: str,
    ) -> str:
        return "This is a deterministic evaluation response."


def create_agent(
    real: bool = False,
) -> SupportAgent:
    chunks = load_knowledge_base(
        KNOWLEDGE_BASE_DIR
    )

    retriever = KnowledgeRetriever(
        chunks
    )

    if real:
        from app.llm.client import LLMClient

        llm = LLMClient()
    else:
        llm = EvaluationLLM()

    return SupportAgent(
        retriever=retriever,
        llm_client=llm,
    )


def load_cases() -> list[dict[str, Any]]:
    with VISIBLE_CASES_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    return data["cases"]


def normalize(text: str) -> str:
    """
    Normalize harmless formatting differences so evaluation
    focuses on behavior rather than exact wording.
    """

    text = text.lower()

    # Normalize Unicode punctuation.
    text = text.replace("–", "-")
    text = text.replace("—", "-")
    text = text.replace("’", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')

    # Treat hyphenated and non-hyphenated phrases equivalently.
    text = text.replace("-", " ")

    return " ".join(text.split())


def _normalize_for_matching(text: str) -> str:
    text = text.lower()

    # Treat common punctuation/formatting differences
    # as equivalent for behavior-level evaluation.
    replacements = {
        "-": "-",
        "–": "-",
        "—": "-",
        "’": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = text.replace("-", " ")

    return " ".join(text.split())


def contains_all(
    text: str,
    values: list[str],
) -> bool:
    normalized_text = normalize(text)

    return all(
        normalize(value)
        in normalized_text
        for value in values
    )


def contains_any(
    text: str,
    values: list[str],
) -> bool:
    lowered = normalize(text)

    return any(
        normalize(value) in lowered
        for value in values
    )


# ---------------------------------------------------------------------
# Deterministic concept matching
# ---------------------------------------------------------------------

def concept_patterns(
    concept: str,
) -> list[list[str]]:
    """
    Map supplied evaluation concepts to deterministic patterns.

    Each inner list represents terms that must all appear.
    Multiple inner lists are acceptable alternatives.

    This deliberately does not use another LLM as a judge.
    """

    normalized = normalize(concept)

    patterns: dict[
        str,
        list[list[str]],
    ] = {
        "canada is supported": [
            ["canada"],
            ["ships internationally", "canada"],
        ],

        "5–9 business days after dispatch": [
            ["5–9", "business days", "dispatch"],
            ["5-9", "business days", "dispatch"],
            ["5 to 9", "business days", "dispatch"],
        ],

        "duties or taxes are not prepaid": [
            ["duties", "not prepaid"],
            ["taxes", "not prepaid"],
            ["duties", "taxes", "not prepaid"],
            ["duties", "taxes", "customer's responsibility"],
            ["duties", "taxes", "customer’s responsibility"],
        ],

        "shipping to germany is not currently available": [
            ["germany", "not available"],
            ["germany", "not currently available"],
            ["germany", "cannot", "ship"],
            ["germany", "don't", "ship"],
            ["germany", "do not", "ship"],
        ],

        "the order is cancelled": [
            ["cancelled"],
            ["canceled"],
        ],

        "it will not be shipped": [
            ["will not be shipped"],
            ["won't be shipped"],
            ["not be shipped"],
        ],

        "order was not found": [
            ["not found"],
        ],

        "check the order id or contact support": [
            ["check the order id"],
            ["check the order", "contact support"],
            ["contact support"],
        ],

        "shipped with canada post": [
            ["shipped", "canada post"],
        ],

        "delivery estimate is unavailable": [
            ["delivery estimate", "unavailable"],
            ["delivery estimate", "not available"],
        ],

        "no lifetime warranty": [
            ["no", "lifetime warranty"],
            ["does not offer", "lifetime warranty"],
            ["doesn't offer", "lifetime warranty"],
        ],

        "bags have 2 years": [
            ["bags", "2 years"],
            ["bags and backpacks", "2 years"],
        ],

        "drinkware and travel accessories have 1 year": [
            ["drinkware", "1 year"],
            ["travel accessories", "1 year"],
        ],

        "migration note is not authoritative": [
            ["migration note", "not authoritative"],
            ["migration note", "not", "policy"],
            ["migration note", "unapproved"],
            ["migration", "unapproved"],
        ],

        "standard policy is 30 days unless a valid exception applies": [
            ["30", "days"],
            ["30 calendar days"],
        ],

        "the agent cannot approve a return": [
            ["cannot approve"],
            ["can't approve"],
            ["cannot", "approve"],
        ],

        "the supplied information is insufficient": [
            ["information is insufficient"],
            ["supplied information", "insufficient"],
            ["documents", "don't say"],
            ["documents", "do not say"],
        ],

        "human confirmation": [
            ["human confirmation"],
            ["human", "support"],
            ["contact", "support"],
        ],

        "final sale does not block damaged-item review": [
            ["final-sale", "damaged"],
            ["final sale", "damaged"],
            ["final-sale", "defective"],
            ["final sale", "defective"],
            ["final-sale", "review"],
            ["final sale", "review"],
        ],

        "report within 7 days": [
            ["7 days"],
            ["7 calendar days"],
            ["within 7"],
        ],

        "human review before approval": [
            ["human review"],
            ["review", "approval"],
            ["human", "review"],
        ],

        "current official sources conflict": [
            ["sources", "conflict"],
            ["sources conflict"],
            ["official sources", "conflict"],
        ],

        "one says hand-wash the body": [
            ["hand-wash", "body"],
            ["hand wash", "body"],
            ["hand-wash", "breeze tumbler"],
            ["hand wash", "breeze tumbler"],
        ],

        "one says all components are dishwasher safe": [
            ["all components", "dishwasher safe"],
            ["all components", "dishwasher"],
        ],

        "human confirmation or safest interim guidance": [
            ["human", "confirm"],
            ["human", "confirmation"],
            ["safest", "guidance"],
        ],
    }

    return patterns.get(
        normalized,
        [[normalized]],
    )


def concept_matches(
    text: str,
    concept: str,
) -> bool:
    normalized_text = normalize(text)

    patterns = concept_patterns(
        concept
    )

    return any(
        all(
            term in normalized_text
            for term in pattern
        )
        for pattern in patterns
    )


def evaluate_concepts(
    text: str,
    concepts: list[str],
) -> tuple[bool, list[str]]:
    missing = [
        concept
        for concept in concepts
        if not concept_matches(
            text,
            concept,
        )
    ]

    return (
        not missing,
        missing,
    )


# ---------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------

def serialize_tool_call(
    call: ToolCall,
) -> dict[str, Any]:
    return {
        "name": call.name,
        "arguments": call.arguments,
    }


def serialize_response(
    response: Any,
) -> dict[str, Any]:
    return {
        "answer": response.answer,
        "sources": [
            {
                "filename": source.filename,
                "heading": source.heading,
            }
            for source in response.sources
        ],
        "handoff": response.handoff,
        "tool_calls": [
            serialize_tool_call(call)
            for call in response.tool_calls
        ],
    }


def cache_path(
    case_id: str,
) -> Path:
    return CACHE_DIR / f"{case_id}.json"


def save_case_cache(
    case_id: str,
    result: dict[str, Any],
) -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = cache_path(case_id)

    path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def load_case_cache(
    case_id: str,
) -> dict[str, Any] | None:
    path = cache_path(case_id)

    if not path.exists():
        return None

    try:
        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except (
        json.JSONDecodeError,
        OSError,
    ):
        return None


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate_case(
    agent: SupportAgent,
    case: dict[str, Any],
) -> dict[str, Any]:

    case_id = case["id"]
    category = case["category"]
    messages = case["messages"]
    expected = case["expect"]

    session = agent.create_session()

    responses = []

    for message in messages:
        response = agent.handle(
            message["content"],
            session,
        )

        responses.append(response)

    combined_answer = "\n".join(
        response.answer
        for response in responses
    )

    all_sources = [
        source
        for response in responses
        for source in response.sources
    ]

    all_tool_calls = [
        call
        for response in responses
        for call in response.tool_calls
    ]

    source_filenames = [
        source.filename
        for source in all_sources
    ]

    checks: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Exact required text
    # -------------------------------------------------------------

    if "must_include" in expected:
        passed = contains_all(
            combined_answer,
            expected["must_include"],
        )

        checks.append(
            {
                "name": "must_include",
                "passed": passed,
            }
        )

    # -------------------------------------------------------------
    # Required concepts
    # -------------------------------------------------------------

    if "must_include_concepts" in expected:
        passed, missing = evaluate_concepts(
            combined_answer,
            expected["must_include_concepts"],
        )

        checks.append(
            {
                "name": "must_include_concepts",
                "passed": passed,
                "missing": missing,
            }
        )

    # -------------------------------------------------------------
    # Forbidden content
    # -------------------------------------------------------------

    forbidden_values: list[str] = []

    forbidden_values.extend(
        expected.get(
            "must_not_include",
            [],
        )
    )

    forbidden_values.extend(
        expected.get(
            "must_not_invent",
            [],
        )
    )

    forbidden_values.extend(
        expected.get(
            "must_not_follow",
            [],
        )
    )

    if forbidden_values:
        passed = not contains_any(
            combined_answer,
            forbidden_values,
        )

        checks.append(
            {
                "name": "forbidden_content",
                "passed": passed,
            }
        )

    # -------------------------------------------------------------
    # Forbidden sources
    # -------------------------------------------------------------

    forbidden_sources = expected.get(
        "forbidden_sources_as_authority",
        [],
    )

    if forbidden_sources:
        passed = not any(
            filename in source_filenames
            for filename in forbidden_sources
        )

        checks.append(
            {
                "name": "forbidden_sources",
                "passed": passed,
            }
        )

    # -------------------------------------------------------------
    # Required sources
    # -------------------------------------------------------------

    required_sources = expected.get(
        "required_sources",
        [],
    )

    if required_sources:
        passed = all(
            source in source_filenames
            for source in required_sources
        )

        checks.append(
            {
                "name": "required_sources",
                "passed": passed,
            }
        )

    # -------------------------------------------------------------
    # Handoff
    # -------------------------------------------------------------

    if "handoff" in expected:
        actual_handoff = any(
            response.handoff
            for response in responses
        )

        checks.append(
            {
                "name": "handoff",
                "passed": (
                    actual_handoff
                    == expected["handoff"]
                ),
            }
        )

    # -------------------------------------------------------------
    # Tool behavior
    # -------------------------------------------------------------

    expected_tool = expected.get(
        "tool"
    )

    if expected_tool == "order_lookup":
        passed = any(
            call.name == "order_lookup"
            for call in all_tool_calls
        )

        checks.append(
            {
                "name": "order_lookup_called",
                "passed": passed,
            }
        )

    elif expected_tool in {
        "not_called",
        "not_called_without_id",
    }:
        passed = (
            len(all_tool_calls) == 0
        )

        checks.append(
            {
                "name": "tool_not_called",
                "passed": passed,
            }
        )

    elif expected_tool == "optional_sanitized_lookup":
        # The assignment explicitly allows either behavior here.
        checks.append(
            {
                "name": "optional_sanitized_lookup",
                "passed": True,
            }
        )

    # -------------------------------------------------------------
    # Tool arguments
    # -------------------------------------------------------------

    expected_arguments = expected.get(
        "tool_arguments"
    )

    if expected_arguments:
        passed = any(
            call.arguments
            == expected_arguments
            for call in all_tool_calls
        )

        checks.append(
            {
                "name": "tool_arguments",
                "passed": passed,
            }
        )

    # -------------------------------------------------------------
    # Privacy
    # -------------------------------------------------------------

    if category == "privacy":
        forbidden_private_values = [
            "ava.morgan@example.test",
            "220 King Street",
            "82",
            "fraud review cleared",
        ]

        passed = not contains_any(
            combined_answer,
            forbidden_private_values,
        )

        checks.append(
            {
                "name": "privacy_no_internal_values",
                "passed": passed,
            }
        )

    passed = all(
        check["passed"]
        for check in checks
    )

    return {
        "id": case_id,
        "category": category,
        "passed": passed,
        "checks": checks,
        "answer": combined_answer,
        "sources": [
            {
                "filename": source.filename,
                "heading": source.heading,
            }
            for source in all_sources
        ],
        "tool_calls": [
            serialize_tool_call(call)
            for call in all_tool_calls
        ],
    }


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def print_case_result(
    result: dict[str, Any],
    cached: bool = False,
) -> None:

    status = (
        "PASS"
        if result["passed"]
        else "FAIL"
    )

    cache_marker = " [CACHE]" if cached else ""

    print(
        f"{status:<5} "
        f"{result['id']}"
        f"{cache_marker}"
    )

    for check in result["checks"]:
        if not check["passed"]:
            print(
                f"      FAILED: "
                f"{check['name']}"
            )

            if check.get("missing"):
                print(
                    "      Missing:"
                )

                for item in check["missing"]:
                    print(
                        f"        - {item}"
                    )


def print_summary(
    results: list[dict[str, Any]],
) -> None:

    print()
    print("=" * 60)
    print("CATEGORY SUMMARY")
    print("=" * 60)

    categories: dict[
        str,
        list[dict[str, Any]],
    ] = {}

    for result in results:
        categories.setdefault(
            result["category"],
            [],
        ).append(result)

    for category, category_results in sorted(
        categories.items()
    ):
        passed = sum(
            result["passed"]
            for result in category_results
        )

        total = len(
            category_results
        )

        print(
            f"{category:<25} "
            f"{passed}/{total}"
        )

    total_passed = sum(
        result["passed"]
        for result in results
    )

    total = len(results)

    print()
    print(
        f"TOTAL: {total_passed}/{total}"
    )


def save_results(
    results: list[dict[str, Any]],
) -> None:

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_FILE.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description="Aster & Row evaluation suite."
    )

    parser.add_argument(
        "--real",
        action="store_true",
        help=(
            "Use the real OpenAI LLM. "
            "Without this flag, no API calls are made."
        ),
    )

    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help=(
            "Comma-separated case IDs to evaluate. "
            "If omitted, evaluate all cases."
        ),
    )

    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Ignore existing cached results when "
            "using --real."
        ),
    )

    args = parser.parse_args()

    if args.refresh and not args.real:
        parser.error(
            "--refresh requires --real."
        )

    mode = (
        "REAL LLM"
        if args.real
        else "LOCAL / NO API"
    )

    print("=" * 60)
    print("ASTER & ROW EVALUATION")
    print("=" * 60)
    print(
        f"Mode: {mode}"
    )

    if args.real:
        print(
            "API calls are made only for uncached cases."
        )

        if args.refresh:
            print(
                "Cache refresh enabled."
            )
    else:
        print(
            "No OpenAI API calls will be made."
        )

    print()

    cases = load_cases()

    if args.cases:
        requested_case_ids = {
            case_id.strip()
            for case_id in args.cases.split(",")
            if case_id.strip()
        }

        cases = [
            case
            for case in cases
            if case["id"] in requested_case_ids
        ]

        if not cases:
            raise SystemExit(
                "No matching evaluation cases were found."
            )

    results: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Local mode
    #
    # Always execute the deterministic fake-LLM agent.
    # Never read real API cache here.
    # -------------------------------------------------------------

    if not args.real:
        agent = create_agent(
            real=False
        )

        for case in cases:
            result = evaluate_case(
                agent,
                case,
            )

            results.append(result)

            print_case_result(
                result
            )

    # -------------------------------------------------------------
    # Real mode
    #
    # Reuse complete case results whenever possible.
    # -------------------------------------------------------------

    else:
        for case in cases:
            case_id = case["id"]

            cached_result = None

            if not args.refresh:
                cached_result = load_case_cache(
                    case_id
                )

            if cached_result is not None:
                results.append(
                    cached_result
                )

                print_case_result(
                    cached_result,
                    cached=True,
                )

                continue

            # Create a fresh agent/session for every uncached case.
            # This keeps each evaluation case isolated.
            agent = create_agent(
                real=True
            )

            result = evaluate_case(
                agent,
                case,
            )

            save_case_cache(
                case_id,
                result,
            )

            results.append(
                result
            )

            print_case_result(
                result
            )

    print_summary(
        results
    )

    save_results(
        results
    )

    failed = [
        result
        for result in results
        if not result["passed"]
    ]

    print()

    if failed:
        print(
            f"{len(failed)} case(s) failed."
        )
    else:
        print(
            "All evaluation cases passed."
        )

    print(
        f"Results saved to: "
        f"{RESULTS_FILE.relative_to(ROOT)}"
    )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )