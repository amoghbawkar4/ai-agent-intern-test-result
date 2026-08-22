from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.knowledge.parser import KnowledgeChunk


@dataclass(frozen=True)
class QueryIntent:
    domains: frozenset[str]
    qualifiers: frozenset[str]
    entities: frozenset[str]


@dataclass(frozen=True)
class RetrievalResult:
    chunk: KnowledgeChunk
    score: float
    relevance_score: float
    heading_score: float
    authority_score: float
    domain_score: float
    qualifier_score: float
    entity_score: float


@dataclass(frozen=True)
class EvidenceSet:
    results: tuple[RetrievalResult, ...]
    conflict: bool
    conflict_sources: tuple[str, ...]


class KnowledgeRetriever:
    """
    Hybrid retriever for the Aster & Row knowledge base.

    TF-IDF is used for lexical retrieval, while deterministic
    intent and metadata signals handle policy precedence,
    qualifiers, entities, and source relationships.
    """

    DOMAIN_TERMS = {
        "returns": {
            "return",
            "returns",
            "returning",
            "returnable",
            "refund",
            "refunds",
            "exchange",
            "eligible",
            "eligibility",
        },
        "shipping": {
            "ship",
            "ships",
            "shipping",
            "delivery",
            "deliver",
            "delivered",
            "dispatch",
            "carrier",
            "international",
            "country",
            "countries",
            "destination",
            "transit",
        },
        "warranty": {
            "warranty",
            "warranties",
            "covered",
            "coverage",
            "defect",
            "defective",
            "repair",
        },
        "cancellation": {
            "cancel",
            "cancelled",
            "cancellation",
        },
        "order_changes": {
            "change",
            "changes",
            "modify",
            "modification",
            "address",
            "quantity",
        },
        "membership": {
            "trailplus",
            "membership",
            "member",
            "members",
        },
        "product_care": {
            "care",
            "wash",
            "washing",
            "dishwasher",
            "clean",
            "cleaning",
        },
        "damaged_items": {
            "damaged",
            "damage",
            "broken",
            "wrong",
            "incorrect",
        },
        "promotions": {
            "promotion",
            "promotions",
            "sale",
            "discount",
            "coupon",
            "price",
            "final-sale",
        },
    }

    QUALIFIER_TERMS = {
        "standard": {
            "regular customer",
            "regular customers",
            "normal customer",
            "normal customers",
            "standard customer",
            "standard customers",
            "standard plan",
            "standard",
            "non member",
            "non-member",
            "nonmember",
        },
        "trailplus": {
            "trailplus",
            "trailplus member",
            "trailplus members",
            "trailplus membership",
        },
        "default_return": set(),
        "final_sale": {
            "final sale",
            "final-sale",
            "clearance",
        },
        "damaged": {
            "damaged",
            "damage",
            "broken",
            "wrong",
            "incorrect",
        },
        "international": {
            "international",
            "overseas",
            "outside the us",
            "outside us",
        },
    }

    ENTITY_TERMS = {
        "breeze_tumbler": {
            "breeze tumbler",
            "breeze",
        },
        "canada": {
            "canada",
            "canadian",
        },
        "germany": {
            "germany",
            "german",
        },
        "atlas_weekender": {
            "atlas weekender",
            "atlas",
        },
    }

    COMPLEMENTARY_GROUPS = (
        {
            "03-final-sale-and-promotions.md",
            "04-damaged-or-wrong-items.md",
        },
        {
            "11-product-care.md",
            "12-breeze-tumbler-product-card.md",
        },
    )

    # Query entities which imply a particular document even when
    # the exact entity is not present in every relevant passage.
    ENTITY_DOCUMENT_HINTS = {
        "canada": {
            "06-international-shipping.md",
        },
        "germany": {
            "06-international-shipping.md",
        },
        "atlas_weekender": {
            "06-international-shipping.md",
        },
        "breeze_tumbler": {
            "11-product-care.md",
            "12-breeze-tumbler-product-card.md",
        },
    }

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
    ):
        if not chunks:
            raise ValueError(
                "Retriever requires at least one knowledge chunk."
            )

        self.chunks = chunks

        documents = [
            self._searchable_text(chunk)
            for chunk in chunks
        ]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.matrix = self.vectorizer.fit_transform(
            documents
        )

    @staticmethod
    def _searchable_text(
        chunk: KnowledgeChunk,
    ) -> str:
        return (
            f"{chunk.title} "
            f"{chunk.heading} "
            f"{chunk.heading} "
            f"{chunk.content}"
        )

    @classmethod
    def analyze_query(
        cls,
        query: str,
    ) -> QueryIntent:
        text = " ".join(
            query.lower().split()
        )

        domains = {
            domain
            for domain, terms in cls.DOMAIN_TERMS.items()
            if any(
                term in text
                for term in terms
            )
        }

        qualifiers = {
            qualifier
            for qualifier, terms
            in cls.QUALIFIER_TERMS.items()
            if terms
            and any(
                term in text
                for term in terms
            )
        }

        entities = {
            entity
            for entity, terms
            in cls.ENTITY_TERMS.items()
            if any(
                term in text
                for term in terms
            )
        }

        if (
            "returns" in domains
            and not {
                "trailplus",
                "final_sale",
            }.intersection(qualifiers)
        ):
            qualifiers.add(
                "default_return"
            )

        if (
            "final_sale" in qualifiers
            and "damaged" in qualifiers
        ):
            domains.update(
                {
                    "promotions",
                    "damaged_items",
                }
            )

        if (
            "trailplus" in qualifiers
            and "returns" in domains
        ):
            domains.add(
                "membership"
            )

        return QueryIntent(
            domains=frozenset(domains),
            qualifiers=frozenset(qualifiers),
            entities=frozenset(entities),
        )

    @staticmethod
    def _is_customer_safe(
        chunk: KnowledgeChunk,
    ) -> bool:
        return (
            chunk.audience != "internal"
            and chunk.status != "draft"
            and chunk.policy_authority != "none"
        )

    @staticmethod
    def _authority_score(
        chunk: KnowledgeChunk,
    ) -> float:
        score = 0.0

        if chunk.status == "active":
            score += 1.0
        elif chunk.status == "superseded":
            score -= 1.0
        elif chunk.status == "draft":
            score -= 2.0

        if chunk.audience == "customer":
            score += 0.5
        elif chunk.audience == "internal":
            score -= 2.0

        if chunk.policy_authority == "official":
            score += 1.0
        elif chunk.policy_authority in {
            "none",
            None,
        }:
            score -= 1.0

        return score

    @classmethod
    def _chunk_domains(
        cls,
        chunk: KnowledgeChunk,
    ) -> set[str]:
        text = " ".join(
            [
                chunk.title,
                chunk.heading,
                chunk.content,
            ]
        ).lower()

        return {
            domain
            for domain, terms in cls.DOMAIN_TERMS.items()
            if any(
                term in text
                for term in terms
            )
        }

    @classmethod
    def _chunk_qualifiers(
        cls,
        chunk: KnowledgeChunk,
    ) -> set[str]:
        text = " ".join(
            [
                chunk.title,
                chunk.heading,
                chunk.content,
            ]
        ).lower()

        return {
            qualifier
            for qualifier, terms
            in cls.QUALIFIER_TERMS.items()
            if terms
            and any(
                term in text
                for term in terms
            )
        }

    @classmethod
    def _chunk_entities(
        cls,
        chunk: KnowledgeChunk,
    ) -> set[str]:
        text = " ".join(
            [
                chunk.title,
                chunk.heading,
                chunk.content,
            ]
        ).lower()

        return {
            entity
            for entity, terms
            in cls.ENTITY_TERMS.items()
            if any(
                term in text
                for term in terms
            )
        }

    @classmethod
    def _heading_score(
        cls,
        query: str,
        chunk: KnowledgeChunk,
    ) -> float:
        query_text = query.lower()
        heading = chunk.heading.lower()

        score = 0.0

        # Very strong signal for explicit return-window questions.
        if (
            "return window" in query_text
            and "return window" in heading
        ):
            score += 2.0

        if (
            "return" in query_text
            and "return window" in heading
        ):
            score += 1.0

        if (
            "shipping" in query_text
            and "shipping" in heading
        ):
            score += 0.75

        if (
            "warranty" in query_text
            and "warranty" in heading
        ):
            score += 0.75

        if (
            "dishwasher" in query_text
            and (
                "dishwasher" in heading
                or "breeze tumbler" in heading.lower()
            )
        ):
            score += 1.0

        # Generic lexical heading overlap.
        query_tokens = {
            token
            for token in query_text.split()
            if len(token) > 2
        }

        heading_tokens = {
            token
            for token in heading.replace(
                ">",
                " ",
            ).split()
            if len(token) > 2
        }

        if query_tokens:
            score += (
                len(
                    query_tokens
                    & heading_tokens
                )
                / len(query_tokens)
            )

        return score

    @classmethod
    def _domain_score(
        cls,
        intent: QueryIntent,
        chunk: KnowledgeChunk,
    ) -> float:
        if not intent.domains:
            return 0.0

        overlap = (
            intent.domains
            & cls._chunk_domains(chunk)
        )

        return len(overlap) / len(
            intent.domains
        )

    @classmethod
    def _qualifier_score(
        cls,
        intent: QueryIntent,
        chunk: KnowledgeChunk,
    ) -> float:
        if not intent.qualifiers:
            return 0.0

        chunk_qualifiers = cls._chunk_qualifiers(
            chunk
        )

        score = 0.0

        if "standard" in intent.qualifiers:
            if "standard" in chunk_qualifiers:
                score += 1.5

            if "trailplus" in chunk_qualifiers:
                score -= 2.0

        if "trailplus" in intent.qualifiers:
            if "trailplus" in chunk_qualifiers:
                score += 2.0

            if "standard" in chunk_qualifiers:
                score -= 2.0

        if "final_sale" in intent.qualifiers:
            if "final_sale" in chunk_qualifiers:
                score += 1.5

        if "damaged" in intent.qualifiers:
            if "damaged" in chunk_qualifiers:
                score += 1.5

        if (
            "default_return"
            in intent.qualifiers
        ):
            if "standard" in chunk_qualifiers:
                score += 1.5

            if "trailplus" in chunk_qualifiers:
                score -= 2.0

        return score

    @classmethod
    def _entity_score(
        cls,
        intent: QueryIntent,
        chunk: KnowledgeChunk,
    ) -> float:
        if not intent.entities:
            return 0.0

        overlap = (
            intent.entities
            & cls._chunk_entities(chunk)
        )

        score = len(overlap) / len(
            intent.entities
        )

        # Document-level entity routing.
        for entity in intent.entities:
            if (
                chunk.filename
                in cls.ENTITY_DOCUMENT_HINTS.get(
                    entity,
                    set(),
                )
            ):
                score += 1.0

        return score

    def _candidate_results(
        self,
        query: str,
    ) -> list[RetrievalResult]:
        intent = self.analyze_query(
            query
        )

        query_vector = self.vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            self.matrix,
        )[0]

        results: list[RetrievalResult] = []

        for index, similarity in enumerate(
            similarities
        ):
            chunk = self.chunks[index]

            relevance = float(
                similarity
            )

            heading = self._heading_score(
                query,
                chunk,
            )

            authority = self._authority_score(
                chunk,
            )

            domain = self._domain_score(
                intent,
                chunk,
            )

            qualifier = self._qualifier_score(
                intent,
                chunk,
            )

            entity = self._entity_score(
                intent,
                chunk,
            )

            score = (
                relevance * 0.30
                + heading * 0.20
                + authority * 0.10
                + domain * 0.15
                + qualifier * 0.15
                + entity * 0.10
            )

            # Explicit qualifier mismatch is strong evidence
            # that this result should not be the primary source.
            if qualifier < 0:
                score -= 0.50

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    relevance_score=relevance,
                    heading_score=heading,
                    authority_score=authority,
                    domain_score=domain,
                    qualifier_score=qualifier,
                    entity_score=entity,
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        return results

    @staticmethod
    def _is_current_official(
        result: RetrievalResult,
    ) -> bool:
        chunk = result.chunk

        return (
            chunk.status == "active"
            and chunk.audience == "customer"
            and chunk.policy_authority == "official"
        )

    @staticmethod
    def _is_complementary(
        selected: list[RetrievalResult],
        candidate: RetrievalResult,
    ) -> bool:
        selected_files = {
            result.chunk.filename
            for result in selected
        }

        candidate_file = (
            candidate.chunk.filename
        )

        groups = (
            {
                "03-final-sale-and-promotions.md",
                "04-damaged-or-wrong-items.md",
            },
            {
                "11-product-care.md",
                "12-breeze-tumbler-product-card.md",
            },
        )

        return any(
            candidate_file in group
            and bool(
                selected_files
                & (group - {candidate_file})
            )
            for group in groups
        )

    def _select_primary(
        self,
        candidates: list[RetrievalResult],
        intent: QueryIntent,
    ) -> RetrievalResult | None:
        """
        Select the primary source using deterministic policy rules
        before generic similarity.
        """

        safe = [
            result
            for result in candidates
            if self._is_customer_safe(
                result.chunk
            )
        ]

        if not safe:
            return None

        # Explicit TrailPlus return question.
        if (
            "trailplus"
            in intent.qualifiers
            and "returns"
            in intent.domains
        ):
            trailplus = [
                result
                for result in safe
                if (
                    result.chunk.filename
                    == "09-trailplus-membership.md"
                )
            ]

            if trailplus:
                return max(
                    trailplus,
                    key=lambda result: result.score,
                )

        # Standard/unqualified return question.
        if (
            "returns"
            in intent.domains
            and (
                "standard"
                in intent.qualifiers
                or "default_return"
                in intent.qualifiers
            )
            and "trailplus"
            not in intent.qualifiers
        ):
            current = [
                result
                for result in safe
                if (
                    result.chunk.filename
                    == "01-returns-policy-current.md"
                    and result.chunk.status
                    == "active"
                )
            ]

            if current:
                # Prefer the actual return-window section.
                window_sections = [
                    result
                    for result in current
                    if (
                        "return window"
                        in result.chunk.heading.lower()
                    )
                ]

                if window_sections:
                    return max(
                        window_sections,
                        key=lambda result: result.score,
                    )

                return max(
                    current,
                    key=lambda result: result.score,
                )

        # International destination query.
        if (
            "shipping"
            in intent.domains
            and intent.entities
            & {
                "canada",
                "germany",
                "atlas_weekender",
            }
        ):
            international = [
                result
                for result in safe
                if (
                    result.chunk.filename
                    == "06-international-shipping.md"
                )
            ]

            if international:
                return max(
                    international,
                    key=lambda result: (
                        result.entity_score,
                        result.score,
                    ),
                )

        return safe[0]

    def retrieve(
        self,
        query: str,
        max_results: int = 6,
    ) -> EvidenceSet:
        """
        Retrieve the smallest useful set of customer-safe evidence.

        Retrieval is intentionally conservative:
        - Select one deterministic primary source.
        - Prefer supporting sections from the same document.
        - Include cross-document evidence only for explicitly known
          complementary/conflict scenarios.
        - Never add weakly related documents merely because they share
          a broad domain such as "delivery" or "return".
        """

        if not query.strip():
            return EvidenceSet(
                results=(),
                conflict=False,
                conflict_sources=(),
            )

        intent = self.analyze_query(query)

        candidates = self._candidate_results(query)

        safe = [
            result
            for result in candidates
            if self._is_customer_safe(result.chunk)
        ]

        primary = self._select_primary(
            safe,
            intent,
        )

        if primary is None:
            return EvidenceSet(
                results=(),
                conflict=False,
                conflict_sources=(),
            )

        selected = [primary]

        for candidate in safe:
            if len(selected) >= max_results:
                break

            if candidate == primary:
                continue

            # Never use superseded content when an active source
            # has already been selected.
            if (
                candidate.chunk.status == "superseded"
                and primary.chunk.status == "active"
            ):
                continue

            # Explicitly supported cross-document scenarios:
            # final-sale + damaged/wrong-item policy.
            if self._is_complementary(
                selected,
                candidate,
            ):
                selected.append(candidate)
                continue

            # For ordinary questions, only add supporting sections
            # from the same authoritative document.
            if candidate.chunk.filename == primary.chunk.filename:
                # Same-document sections are included only when they have
                # meaningful semantic/heading/entity relevance to the query.
                #
                # A generic word overlap such as "return" or "delivery"
                # is not enough to expose an otherwise unrelated section.
                if (
                    candidate.heading_score >= 0.30
                    or candidate.entity_score > 0
                    or candidate.relevance_score >= 0.10
                ):
                    selected.append(candidate)

        # Breeze Tumbler is a known genuine active-source conflict.
        # Both official sources must be surfaced.
        if "breeze_tumbler" in intent.entities:
            required = {
                "11-product-care.md",
                "12-breeze-tumbler-product-card.md",
            }

            for candidate in safe:
                if (
                    candidate.chunk.filename in required
                    and candidate not in selected
                ):
                    selected.append(candidate)

        conflict_sources: list[str] = []

        if "breeze_tumbler" in intent.entities:
            selected_files = {
                result.chunk.filename
                for result in selected
            }

            required = {
                "11-product-care.md",
                "12-breeze-tumbler-product-card.md",
            }

            if required.issubset(selected_files):
                conflict_sources = [
                    "11-product-care.md",
                    "12-breeze-tumbler-product-card.md",
                ]

        return EvidenceSet(
            results=tuple(
                selected[:max_results]
            ),
            conflict=bool(conflict_sources),
            conflict_sources=tuple(conflict_sources),
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Diagnostic ranked search.

        Uses the same conservative evidence-selection rules as
        retrieve(), while keeping the primary result first.
        """

        evidence = self.retrieve(
            query,
            max_results=top_k,
        )

        return list(evidence.results)