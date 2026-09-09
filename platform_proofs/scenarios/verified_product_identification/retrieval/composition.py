"""Scenario-owned composition root for retrieval adapters."""

from __future__ import annotations

from platform_proofs.scenarios.verified_product_identification.application.ports.catalog_search import (
    ExactIdentifierLookupPort,
    LexicalCandidateSearchPort,
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
