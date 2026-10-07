# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project execution effective profile provenance for reconstruction scope."""

from __future__ import annotations

from intergrax.contracts.execution_effective_profile_provenance import (
    ExecutionEffectiveProfileProvenance,
    ExecutionEffectiveProfileProvenanceReader,
    require_tenant_id_for_profile_provenance,
    validate_execution_effective_profile_provenance_record,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstructionIntegrityError,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent


def discover_execution_ids_in_positioned_history(
    positioned: tuple[PositionedRuntimeEvent, ...],
) -> tuple[ExecutionId, ...]:
    """Collect distinct execution IDs present in positioned runtime history."""
    seen: set[str] = set()
    ordered: list[ExecutionId] = []
    for row in positioned:
        execution_id = validate_execution_id(row.event.execution_id)
        key = str(execution_id)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(execution_id)
    return tuple(sorted(ordered, key=str))


def project_execution_effective_profile_provenance(
    positioned: tuple[PositionedRuntimeEvent, ...],
    *,
    tenant_id: str,
    profile_reader: ExecutionEffectiveProfileProvenanceReader,
) -> tuple[ExecutionEffectiveProfileProvenance, ...]:
    """Fail closed when any in-scope execution lacks an exact tenant binding."""
    tenant = require_tenant_id_for_profile_provenance(tenant_id)
    execution_ids = discover_execution_ids_in_positioned_history(positioned)
    projected: list[ExecutionEffectiveProfileProvenance] = []
    for execution_id in execution_ids:
        record = profile_reader.read(tenant_id=tenant, execution_id=execution_id)
        if record is None:
            raise ExecutionReconstructionIntegrityError(
                "missing effective profile execution binding for reconstruction scope"
            )
        validate_execution_effective_profile_provenance_record(
            record,
            expected_tenant_id=tenant,
            expected_execution_id=execution_id,
        )
        projected.append(record)
    return tuple(projected)


__all__ = [
    "discover_execution_ids_in_positioned_history",
    "project_execution_effective_profile_provenance",
]
