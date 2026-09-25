# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""RagProfile wiring validation at bootstrap (M-RAG.63)."""

from __future__ import annotations

from typing import Optional

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.integration_profile import IntegrationProfile
from intergrax.rag.profiles.rag_profile import (
    APPROVED_PRODUCTION_GRAPH_STORE_SLUGS,
    RagProfile,
    validate_graph_rag_production_wiring,
)

_REGISTERED_RETRIEVER_IDS: frozenset[str] = frozenset(
    {
        "vector_similarity",
        "hybrid",
        "mmr",
        "parent_child",
        "multiquery",
        "hierarchical",
        "fusion",
        "graph_rag",
    }
)

_REGISTERED_CHUNKING_STRATEGIES: frozenset[str] = frozenset(
    {
        "recursive",
        "langchain_recursive",
        "semantic",
        "parent_child",
        "docling",
    }
)


def validate_rag_profile_wiring(
    profile: RagProfile,
    *,
    integration_profile: Optional[IntegrationProfile] = None,
    llm_available: bool = False,
    graph_store_slug: str | None = None,
    production_host: bool = False,
    strict: bool = False,
) -> list[str]:
    """
    Return human-readable wiring issues for ``profile``.

    When ``strict`` is True, unknown retriever/chunking ids are errors; otherwise warnings.
    """
    issues: list[str] = []
    prefix = "error" if strict else "warn"

    def _add(kind: str, message: str) -> None:
        issues.append(f"{kind}:{message}")

    for field_name, retriever_id in (
        ("retriever_id", profile.retriever_id),
        ("fast_retriever_id", profile.fast_retriever_id),
        ("deep_retriever_id", profile.deep_retriever_id),
    ):
        if retriever_id not in _REGISTERED_RETRIEVER_IDS:
            _add(prefix, f"unknown_retriever_id:{field_name}={retriever_id}")

    if profile.chunking_strategy_id not in _REGISTERED_CHUNKING_STRATEGIES:
        _add(prefix, f"unknown_chunking_strategy_id:{profile.chunking_strategy_id}")

    if profile.contextual_enrich == "on" and not llm_available:
        _add("error", "contextual_enrich_requires_llm_adapter")

    if profile.query_expansion == "llm" and not llm_available:
        _add("error", "query_expansion_llm_requires_llm_adapter")

    if profile.agentic_query_mode == "llm" and profile.agentic_enabled and not llm_available:
        _add("error", "agentic_query_llm_requires_llm_adapter")

    if profile.graph_indexer_mode in {"llm", "heuristic_then_llm", "community_report"} and not llm_available:
        _add("warn", f"graph_indexer_mode_{profile.graph_indexer_mode}_works_best_with_llm")

    if profile.graph_rag_enabled and graph_store_slug is None:
        _add("warn", "graph_rag_inmemory_harness_only")

    if profile.uses_hierarchical_index() and not profile.hierarchical_index_enabled:
        _add("warn", "hierarchical_retriever_without_hierarchical_index_flag")

    graph_slug = graph_store_slug
    if integration_profile is not None and graph_slug is None:
        graph_slug = integration_profile.slug_for_category(IntegrationCategory.GRAPH_STORE)
        if graph_slug is None:
            graph_binding = integration_profile.binding_for_field("graph_store")
            if graph_binding is not None:
                graph_slug = graph_binding.resolved_slug()

    if production_host:
        graph_error = validate_graph_rag_production_wiring(profile, graph_store_slug=graph_slug)
        if graph_error is not None:
            _add("error", graph_error)

    if (
        profile.graph_rag_enabled
        and graph_slug is not None
        and graph_slug not in APPROVED_PRODUCTION_GRAPH_STORE_SLUGS
    ):
        _add("warn", f"integration_graph_store_not_in_prod_list:{graph_slug}")

    if profile.sync_ingest_max_bytes <= 0:
        _add("warn", "sync_ingest_max_bytes_disabled_unbounded_sync_path")

    if profile.semantic_chunking_max_chars <= 0 and profile.chunking_strategy_id == "semantic":
        _add("warn", "semantic_chunking_without_size_guard")

    return issues


def assert_rag_profile_wiring(
    profile: RagProfile,
    *,
    integration_profile: Optional[IntegrationProfile] = None,
    llm_available: bool = False,
    graph_store_slug: str | None = None,
) -> None:
    """Raise ``ValueError`` when any error-level issue is detected."""
    issues = validate_rag_profile_wiring(
        profile,
        integration_profile=integration_profile,
        llm_available=llm_available,
        graph_store_slug=graph_store_slug,
        strict=True,
    )
    errors = [issue for issue in issues if issue.startswith("error:")]
    if errors:
        raise ValueError("; ".join(errors))
