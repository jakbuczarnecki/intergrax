# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Project execution integration configuration provenance for reconstruction scope."""

from __future__ import annotations

from intergrax.contracts.execution_event_position import AsOfBoundary
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
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.integrations.contracts.execution_integration_configuration_pinning import (
    ExecutionIntegrationConfigurationPinningError,
)
from intergrax.runtime.events.payload_registry import (
    RuntimeEventPayloadError,
    UnknownPayloadSchemaError,
    validate_payload_envelope,
)
from intergrax.runtime.events.payloads.spine_families import (
    IntegrationConfigurationProvenanceRequirementPayloadV1,
)
from intergrax.runtime.events.spine_payload_codec import legacy_spine_payload_to_typed
from intergrax.runtime.observability.reconstruction.profile_provenance_projection import (
    discover_execution_ids_in_positioned_history,
)


def discover_integration_configuration_provenance_required_execution_ids(
    positioned: tuple[PositionedRuntimeEvent, ...],
) -> frozenset[ExecutionId]:
    """Execution IDs with typed requirement spine evidence in positioned history."""
    required: set[str] = set()
    ordered: list[ExecutionId] = []
    for row in positioned:
        event = row.event
        if event.event_type is not (
            RuntimeEventType.INTEGRATION_CONFIGURATION_PROVENANCE_REQUIREMENT_COMMITTED
        ):
            continue
        execution_id = validate_execution_id(event.execution_id)
        key = str(execution_id)
        if key in required:
            continue
        required.add(key)
        ordered.append(execution_id)
    return frozenset(ordered)


def _decode_requirement_payload(
    event_type: RuntimeEventType,
    payload: dict[str, object],
) -> IntegrationConfigurationProvenanceRequirementPayloadV1:
    if payload.get("payload_schema_id") is not None:
        try:
            typed_envelope = validate_payload_envelope(payload)
        except (RuntimeEventPayloadError, UnknownPayloadSchemaError) as exc:
            raise ExecutionReconstructionIntegrityError(
                "integration configuration requirement payload validation failed",
            ) from exc
        if typed_envelope is None or not isinstance(
            typed_envelope,
            IntegrationConfigurationProvenanceRequirementPayloadV1,
        ):
            raise ExecutionReconstructionIntegrityError(
                "integration configuration requirement typed payload missing",
            )
        return typed_envelope
    typed, _promote = legacy_spine_payload_to_typed(event_type, payload)
    if not isinstance(typed, IntegrationConfigurationProvenanceRequirementPayloadV1):
        raise ExecutionReconstructionIntegrityError(
            "integration configuration requirement payload decode failed",
        )
    return typed


def project_execution_integration_configuration_provenance(
    positioned: tuple[PositionedRuntimeEvent, ...],
    *,
    tenant_id: str,
    reader: ExecutionIntegrationConfigurationProvenanceReader,
    execution_as_of: AsOfBoundary | None = None,
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
    if execution_as_of is not None:
        return (), ExecutionIntegrationConfigurationProvenanceReadStatus.UNAVAILABLE_AT_EXECUTION_BOUNDARY
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
    if required_ids and not projected:
        return (), ExecutionIntegrationConfigurationProvenanceReadStatus.REQUIRED_MISSING
    return tuple(projected), ExecutionIntegrationConfigurationProvenanceReadStatus.CONFIGURED


__all__ = [
    "discover_integration_configuration_provenance_required_execution_ids",
    "project_execution_integration_configuration_provenance",
]
