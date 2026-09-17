# © Artur Czarnecki. All rights reserved.

"""Tier-3 bridge to canonical federated runtime inspection read models."""

from __future__ import annotations

from intergrax.contracts.execution_identity import ExecutionId
from intergrax.contracts.runtime_inspection import (
    RuntimeInspectionQuery,
    RuntimeInspectionReadPort,
    RuntimeInspectionSnapshot,
)


def inspect_execution_runtime_snapshot(
    read_port: RuntimeInspectionReadPort,
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    timeline_limit: int | None = None,
) -> RuntimeInspectionSnapshot:
    """Delegate execution-scope inspection to the canonical runtime read port."""
    query_kwargs: dict[str, object] = {
        "tenant_id": tenant_id,
        "execution_id": execution_id,
    }
    if timeline_limit is not None:
        query_kwargs["timeline_limit"] = timeline_limit
    query = RuntimeInspectionQuery(**query_kwargs)
    return read_port.inspect(query)


__all__ = ["inspect_execution_runtime_snapshot"]
