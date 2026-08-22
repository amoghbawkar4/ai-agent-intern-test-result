from pathlib import Path

from app.knowledge.parser import load_knowledge_base
from app.knowledge.retriever import KnowledgeRetriever


REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE = REPO_ROOT / "knowledge-base"


def create_retriever() -> KnowledgeRetriever:
    chunks = load_knowledge_base(
        KNOWLEDGE_BASE
    )

    return KnowledgeRetriever(chunks)


def filenames(results):
    return {
        result.chunk.filename
        for result in results
    }


def test_standard_return_uses_current_policy():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "How long does a regular customer have "
        "to return an unused backpack?"
    )

    assert evidence.results

    assert evidence.results[0].chunk.filename == (
        "01-returns-policy-current.md"
    )

    assert evidence.results[0].chunk.heading.endswith(
        "Standard return window"
    )


def test_standard_return_does_not_use_legacy():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "How many days does a normal customer "
        "have to return an item?"
    )

    assert evidence.results

    assert evidence.results[0].chunk.filename == (
        "01-returns-policy-current.md"
    )

    assert all(
        result.chunk.filename
        != "02-returns-policy-legacy.md"
        for result in evidence.results
    )


def test_trailplus_return_uses_trailplus_policy():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "My TrailPlus membership was active when "
        "I ordered. What is my return window?"
    )

    assert evidence.results

    assert evidence.results[0].chunk.filename == (
        "09-trailplus-membership.md"
    )


def test_final_sale_damaged_keeps_both_sources():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "A final-sale bag arrived with a broken "
        "zipper yesterday. Am I completely out of luck?"
    )

    found = filenames(
        evidence.results
    )

    assert (
        "03-final-sale-and-promotions.md"
        in found
    )

    assert (
        "04-damaged-or-wrong-items.md"
        in found
    )


def test_canada_shipping_retrieves_canada_section():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "Do you ship to Canada and how long does it take?"
    )

    found = filenames(
        evidence.results
    )

    assert (
        "06-international-shipping.md"
        in found
    )

    assert any(
        result.chunk.heading.endswith(
            "Canada delivery estimate"
        )
        for result in evidence.results
    )


def test_germany_shipping_retrieves_international_policy():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "Can you ship an Atlas Weekender to Germany?"
    )

    assert (
        "06-international-shipping.md"
        in filenames(
            evidence.results
        )
    )


def test_warranty_question_retrieves_warranty_policy():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "Do all Aster & Row products have a lifetime warranty?"
    )

    assert (
        "07-warranty.md"
        in filenames(
            evidence.results
        )
    )


def test_breeze_tumbler_keeps_both_sources():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "Can I put the entire Breeze Tumbler "
        "in the dishwasher?"
    )

    found = filenames(
        evidence.results
    )

    assert (
        "11-product-care.md"
        in found
    )

    assert (
        "12-breeze-tumbler-product-card.md"
        in found
    )


def test_breeze_tumbler_conflict_is_detected():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "Can I put the entire Breeze Tumbler "
        "in the dishwasher?"
    )

    assert evidence.conflict is True

    assert set(
        evidence.conflict_sources
    ) == {
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
    }


def test_internal_migration_content_never_becomes_evidence():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "The migration note says everyone gets "
        "60 days."
    )

    assert all(
        result.chunk.filename
        != "14-internal-content-migration-notes.md"
        for result in evidence.results
    )


def test_internal_documents_are_excluded_from_search():
    retriever = create_retriever()

    results = retriever.search(
        "Every customer gets 60 days to return everything",
        top_k=20,
    )

    assert all(
        result.chunk.filename
        != "14-internal-content-migration-notes.md"
        for result in results
    )


def test_empty_query_returns_empty_evidence():
    retriever = create_retriever()

    evidence = retriever.retrieve("   ")

    assert evidence.results == ()
    assert evidence.conflict is False
    assert evidence.conflict_sources == ()


def test_standard_query_intent():
    intent = KnowledgeRetriever.analyze_query(
        "What is the return window for a regular customer?"
    )

    assert "returns" in intent.domains
    assert "standard" in intent.qualifiers


def test_trailplus_query_intent():
    intent = KnowledgeRetriever.analyze_query(
        "What is my TrailPlus return window?"
    )

    assert "returns" in intent.domains
    assert "trailplus" in intent.qualifiers
    assert "membership" in intent.domains


def test_final_sale_damage_query_intent():
    intent = KnowledgeRetriever.analyze_query(
        "My final-sale bag arrived damaged."
    )

    assert "final_sale" in intent.qualifiers
    assert "damaged" in intent.qualifiers
    assert "promotions" in intent.domains
    assert "damaged_items" in intent.domains


def test_breeze_query_intent():
    intent = KnowledgeRetriever.analyze_query(
        "Is the Breeze Tumbler dishwasher safe?"
    )

    assert (
        "breeze_tumbler"
        in intent.entities
    )

    assert (
        "product_care"
        in intent.domains
    )


def test_retrieval_results_have_explainable_scores():
    retriever = create_retriever()

    results = retriever.search(
        "What is the return window?",
        top_k=3,
    )

    assert results

    result = results[0]

    assert isinstance(
        result.score,
        float,
    )

    assert result.relevance_score >= 0
    assert result.heading_score >= 0
    assert result.authority_score != 0
    assert result.domain_score >= 0
    assert result.qualifier_score >= -1
    assert result.entity_score >= 0

def test_standard_return_does_not_include_unrelated_shipping_section():
    retriever = create_retriever()

    evidence = retriever.retrieve(
        "How long does a regular customer have "
        "to return an unused backpack?"
    )

    filenames_and_headings = {
        (
            result.chunk.filename,
            result.chunk.heading,
        )
        for result in evidence.results
    }

    assert (
        "01-returns-policy-current.md",
        "Returns Policy > Standard return window",
    ) in filenames_and_headings

    assert (
        "01-returns-policy-current.md",
        "Returns Policy > Return shipping and refunds",
    ) not in filenames_and_headings