"""Fail-closed mapping from logical relational targets to physical PostgreSQL objects."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.configuration import (
    DEFAULT_RELATIONAL_TABLE_NAME,
    PostgreSqlBootstrapConfiguration,
    validate_table_identifier,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.postgresql.errors import (
    PostgreSqlBootstrapConfigurationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    RelationalTargetId,
)

_APPROVED_LOGICAL_TARGETS: dict[str, str] = {
    "vpi-products": DEFAULT_RELATIONAL_TABLE_NAME,
}


@dataclass(frozen=True, slots=True)
class PhysicalRelationalTarget:
    schema_name: str
    table_name: str


def resolve_physical_target(
    logical_target: RelationalTargetId,
    configuration: PostgreSqlBootstrapConfiguration,
) -> PhysicalRelationalTarget:
    logical_key = str(logical_target).strip()
    mapped_table = _APPROVED_LOGICAL_TARGETS.get(logical_key)
    if mapped_table is None:
        raise PostgreSqlBootstrapConfigurationError(
            f"unmapped relational target: {logical_key}"
        )
    if mapped_table != configuration.table_name:
        raise PostgreSqlBootstrapConfigurationError(
            f"relational target {logical_key} requires table {mapped_table!r}, "
            f"but configuration specifies {configuration.table_name!r}"
        )
    return PhysicalRelationalTarget(
        schema_name=configuration.schema_name,
        table_name=validate_table_identifier(mapped_table),
    )


def reject_unsafe_logical_target(logical_target: str) -> None:
    candidate = logical_target.strip()
    if candidate in _APPROVED_LOGICAL_TARGETS:
        return
    if ";" in candidate or "--" in candidate or "'" in candidate:
        raise PostgreSqlBootstrapConfigurationError(
            f"unsafe relational target rejected: {logical_target!r}"
        )
    raise PostgreSqlBootstrapConfigurationError(
        f"unmapped relational target: {logical_target!r}"
    )
