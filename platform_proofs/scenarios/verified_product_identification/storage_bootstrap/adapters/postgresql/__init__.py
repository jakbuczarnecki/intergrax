"""PostgreSQL provider adapter for VPI relational storage bootstrap."""

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_IDENTIFIER_TABLE_NAME,
    DEFAULT_RELATIONAL_TABLE_NAME,
    PostgreSqlBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.exact_identifier_lookup import (
    PostgreSqlExactIdentifierLookupAdapter,
)

__all__ = (
    "DEFAULT_IDENTIFIER_TABLE_NAME",
    "DEFAULT_RELATIONAL_TABLE_NAME",
    "PostgreSqlBootstrapConfiguration",
    "PostgreSqlExactIdentifierLookupAdapter",
    "PostgreSqlRelationalStorageAdapter",
)
