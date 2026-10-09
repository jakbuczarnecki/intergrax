# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project execution integration configuration provenance for reconstruction scope."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_integration_configuration_provenance import (
    ExecutionIntegrationConfigurationProvenance,
    ExecutionIntegrationConfigurationProvenanceReadStatus,
    ExecutionIntegrationConfigurationProvenanceReader,
    require_tenant_id_for_integration_configuration_provenance,
    validate_execution_integration_configuration_provenance_record,
)
from intergrax.contracts.execution_reconstruction_models import (
    ExecutionReconstructionIntegrityError,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
)
from intergrax.runtime.observability.reconstruction.profile_provenance_projection import (
    discover_execution_ids_in_positioned_history,
)

_PROVENANCE_REQUIRED_PAYLOAD_KEY = (
    "execution_integration_configuration_provenance_required"
)


def discover_integration_configuration_provenance_required_execution_ids(
    positioned: tuple[PositionedRuntimeEvent, ...],
) -> frozenset[ExecutionId]:
    """
    Execution IDs for which reconstruction requires persisted integration-config provenance.

    Classified only from canonical runtime evidence (typed payload flag) — no heuristic join.
    """
    required: set[str] = set()
    ordered: list[ExecutionId] = []
    for row in positioned:
        payload = row.event.payload
        if type(payload) is not dict:
            continue
        if payload.get(_PROVENANCE_REQUIRED_PAYLOAD_KEY) is not True:
            continue
        execution_id = validate_execution_id(row.event.execution_id)
        key = str(execution_id)
        if key in required:
            continue
        required.add(key)
        ordered.append(execution_id)
    return frozenset(ordered)


def project_execution_integration_configuration_provenance(
    positioned: tuple[PositionedRuntimeEvent, ...],
    *,
    tenant_id: str,
    reader: ExecutionIntegrationConfigurationProvenanceReader,
) -> tuple[
    tuple[ExecutionIntegrationConfigurationProvenance, ...],
    ExecutionIntegrationConfigurationProvenanceReadStatus,
]:
    """Aggregate durable provenance for in-scope execution IDs; fail closed on integrity violations."""
    tenant = require_tenant_id_for_integration_configuration_provenance(tenant_id)
    execution_ids = discover_execution_ids_in_positioned_history(positioned)
    required_ids = discover_integration_configuration_provenance_required_execution_ids(
        positioned,
    )
    projected: list[ExecutionIntegrationConfigurationProvenance] = []
    for execution_id in execution_ids:
        try:
            records = reader.read_all(
                tenant_id=tenant,
                execution_id=execution_id,
            )
        except ExecutionIntegrationConfigurationPinningError as exc:
            raise ExecutionReconstructionIntegrityError(str(exc)) from exc
        if execution_id in required_ids and not records:
            raise ExecutionReconstructionIntegrityError(
                "required integration configuration provenance missing for reconstruction scope",
            )
        for record in records:
            try:
                validate_execution_integration_configuration_provenance_record(
                    record,
                    expected_tenant_id=tenant,
                    expected_execution_id=execution_id,
                )
            except ValueError as exc:
                raise ExecutionReconstructionIntegrityError(str(exc)) from exc
            projected.append(record)
    return tuple(projected), ExecutionIntegrationConfigurationProvenanceReadStatus.CONFIGURED


__all__ = [
    "discover_integration_configuration_provenance_required_execution_ids",
    "project_execution_integration_configuration_provenance",
]
