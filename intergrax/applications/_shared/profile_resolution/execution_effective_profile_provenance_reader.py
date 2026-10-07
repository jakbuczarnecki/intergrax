# © Artur Czarnecki. All rights reserved.

"""Profile Resolution adapter for neutral execution effective profile provenance reads."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionPinningStore,
)
from intergrax.contracts.effective_profile_revision_provenance_ref import (
    EffectiveProfileRevisionProvenanceRef,
)
from intergrax.contracts.execution_effective_profile_provenance import (
    ExecutionEffectiveProfileProvenance,
    ExecutionEffectiveProfileProvenanceReader,
    require_tenant_id_for_profile_provenance,
)
from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id


@dataclass(frozen=True, slots=True)
class PinningStoreExecutionEffectiveProfileProvenanceReader(
    ExecutionEffectiveProfileProvenanceReader,
):
    """Projects ``EffectiveProfileExecutionPinningStore`` bindings into neutral provenance."""

    pinning_store: EffectiveProfileExecutionPinningStore

    def read(
        self,
        *,
        tenant_id: str,
        execution_id: ExecutionId,
    ) -> ExecutionEffectiveProfileProvenance | None:
        tenant = require_tenant_id_for_profile_provenance(tenant_id)
        validated_execution_id = validate_execution_id(execution_id)
        binding = self.pinning_store.get(
            tenant_id=tenant,
            execution_id=validated_execution_id,
        )
        if binding is None:
            return None
        if binding.tenant_id != tenant:
            return None
        if validate_execution_id(binding.execution_id) != validated_execution_id:
            return None
        return ExecutionEffectiveProfileProvenance(
            tenant_id=binding.tenant_id,
            execution_id=binding.execution_id,
            revision_ref=EffectiveProfileRevisionProvenanceRef(binding.revision_id.value),
            fingerprint=binding.fingerprint,
        )


__all__ = ["PinningStoreExecutionEffectiveProfileProvenanceReader"]
