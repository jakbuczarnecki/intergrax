"""Scenario-owned composition root for retrieval adapters."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    ExactIdentifierLookupPort,
    LexicalCandidateSearchPort,
    StructuredCandidateSearchPort,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.exact_identifier_lookup import (
    PostgreSqlExactIdentifierLookupAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.lexical_search_adapter import (
    PostgreSqlLexicalCandidateSearchAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.structured_search_adapter import (
    PostgreSqlStructuredCandidateSearchAdapter,
)


def build_postgresql_exact_identifier_lookup(
    *,
    schema_name: str,
    configuration: PostgreSqlBootstrapConfiguration | None = None,
) -> ExactIdentifierLookupPort:
    """Construct the PostgreSQL exact lookup port for one tenant schema."""
    if configuration is not None:
        return PostgreSqlExactIdentifierLookupAdapter.from_configuration(configuration)
    return PostgreSqlExactIdentifierLookupAdapter.from_env(schema_name=schema_name)


def build_lexical_candidate_search(
    *,
    schema_name: str,
    configuration: PostgreSqlBootstrapConfiguration | None = None,
) -> LexicalCandidateSearchPort:
    """Construct the indexed BM25 lexical search port for one tenant schema."""
    if configuration is not None:
        return PostgreSqlLexicalCandidateSearchAdapter.from_configuration(configuration)
    return PostgreSqlLexicalCandidateSearchAdapter.from_env(schema_name=schema_name)


def build_structured_candidate_search(
    *,
    schema_name: str,
    configuration: PostgreSqlBootstrapConfiguration | None = None,
) -> StructuredCandidateSearchPort:
    """Construct the indexed structured attribute search port for one tenant schema."""
    if configuration is not None:
        return PostgreSqlStructuredCandidateSearchAdapter.from_configuration(configuration)
    return PostgreSqlStructuredCandidateSearchAdapter.from_env(schema_name=schema_name)
