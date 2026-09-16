# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Discovery correlation resolution for machine acquisition (ME-12-C1)."""

from __future__ import annotations

from uuid import uuid4

from intergrax.contracts.marketplace.acquisition import MachineCapabilityAcquisitionRequest
from intergrax.marketplace.acquisition.errors import MachineCapabilityAcquisitionSelectionError


def resolve_acquisition_discovery_correlation(
    request: MachineCapabilityAcquisitionRequest,
    *,
    operation_discovery_correlation_id: str | None = None,
) -> tuple[str, str | None]:
    """Resolve discovery and query correlation for one acquisition operation."""
    if request.observation is not None:
        discovery_correlation_id = request.observation.discovery_correlation_id
        if (
            operation_discovery_correlation_id is not None
            and operation_discovery_correlation_id != discovery_correlation_id
        ):
            raise MachineCapabilityAcquisitionSelectionError(
                "selection discovery_correlation_id must match acquisition observation",
            )
        return discovery_correlation_id, request.observation.query_correlation_id
    if operation_discovery_correlation_id is not None:
        return operation_discovery_correlation_id, None
    return f"machine-acquire-{uuid4()}", None


__all__ = ["resolve_acquisition_discovery_correlation"]
