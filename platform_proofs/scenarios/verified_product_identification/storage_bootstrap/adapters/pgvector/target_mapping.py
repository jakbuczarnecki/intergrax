"""Fail-closed mapping from logical vector targets to physical pgvector tables."""

from __future__ import annotations

from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.configuration import (
    DEFAULT_LOGICAL_TABLE_NAME,
    DEFAULT_LOGICAL_TARGET_NAME,
    PgVectorBootstrapConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.adapters.pgvector.errors import (
    PgVectorBootstrapConfigurationError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    VectorTargetId,
)

_APPROVED_LOGICAL_TARGETS: dict[str, str] = {
    DEFAULT_LOGICAL_TARGET_NAME: DEFAULT_LOGICAL_TABLE_NAME,
}


@dataclass(frozen=True, slots=True)
class PhysicalPgVectorTarget:
    schema_name: str
    table_name: str


def resolve_physical_target(
    logical_target: VectorTargetId,
    configuration: PgVectorBootstrapConfiguration,
) -> PhysicalPgVectorTarget:
    logical_key = str(logical_target).strip()
    mapped_table = _APPROVED_LOGICAL_TARGETS.get(logical_key)
    if mapped_table is None:
        raise PgVectorBootstrapConfigurationError(
            f"unmapped vector target: {logical_key}"
        )
    if mapped_table != configuration.table_name:
        raise PgVectorBootstrapConfigurationError(
            f"vector target {logical_key} requires table {mapped_table!r}, "
            f"but configuration specifies {configuration.table_name!r}"
        )
    return PhysicalPgVectorTarget(
        schema_name=configuration.schema_name,
        table_name=configuration.table_name,
    )


def reject_unsafe_logical_target(logical_target: str) -> None:
    candidate = logical_target.strip()
    if candidate in _APPROVED_LOGICAL_TARGETS:
        return
    if ";" in candidate or "--" in candidate or "'" in candidate:
        raise PgVectorBootstrapConfigurationError(
            f"unsafe vector target rejected: {logical_target!r}"
        )
    raise PgVectorBootstrapConfigurationError(
        f"unmapped vector target: {logical_target!r}"
    )
