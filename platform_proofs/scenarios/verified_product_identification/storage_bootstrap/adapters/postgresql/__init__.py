"""PostgreSQL provider adapter for VPI relational storage bootstrap."""

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.adapter import (
    PostgreSqlRelationalStorageAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_RELATIONAL_TABLE_NAME,
    PostgreSqlBootstrapConfiguration,
)

__all__ = (
    "DEFAULT_RELATIONAL_TABLE_NAME",
    "PostgreSqlBootstrapConfiguration",
    "PostgreSqlRelationalStorageAdapter",
)
