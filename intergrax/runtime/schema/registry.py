# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical runtime schema versions (§42.29, Appendix B.07)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet


@dataclass(frozen=True)
class RuntimeVersionInfo:
    """Compatibility bundle exposed to Tier-3 hosts and plugins."""

    runtime_semver: str = "0.1.0"
    contract_bundle: str = "uaep-1.0"
    supported_schemas: FrozenSet[str] = field(default_factory=frozenset)


# Authoritative list of persisted / wire-format schema ids (increment on breaking change).
RUNTIME_SCHEMA_REGISTRY: dict[str, str] = {
    "agent_decision": "agent_decision.v1",
    "human_request": "human_request.v2",
    "runtime_event": "runtime_event.v1",
    "runtime_checkpoint": "runtime_checkpoint.v1",
    "task_checkpoint": "task_checkpoint.v1",
    "governance_resolution": "governance_resolution.v1",
    "execution_interrupt": "execution_interrupt.v1",
    "policy_decision": "policy_decision.v1",
    "handoff": "handoff.v1",
    "agent_step": "agent_step.v1",
    "pause_record": "pause_record.v1",
    "partial_result": "partial_result.v1",
    "scheduled_resume": "scheduled_resume.v1",
    "shared_task_context": "shared_task_context.v1",
    "agent_context_bundle": "agent_context_bundle.v2",
    "task_context_assembly": "task_context_assembly.v1",
    "task_memory": "task_memory.v1",
    "validation_contract": "validation_contract.v1",
    "nexus_task_worker": "nexus_task_worker.v1",
}

# Post-publication preview schema ids accepted by conformance gates (OBS-EVOL-9.9).
PREVIEW_RUNTIME_SCHEMA_VERSIONS: dict[str, frozenset[str]] = {
    "runtime_event": frozenset({"runtime_event.v2"}),
}


def current_runtime_version() -> RuntimeVersionInfo:
    preview = {v for versions in PREVIEW_RUNTIME_SCHEMA_VERSIONS.values() for v in versions}
    return RuntimeVersionInfo(
        supported_schemas=frozenset(RUNTIME_SCHEMA_REGISTRY.values()) | preview,
    )


def validate_schema_version(schema_id: str, version: str) -> bool:
    expected = RUNTIME_SCHEMA_REGISTRY.get(schema_id)
    if expected is None:
        return False
    if version == expected:
        return True
    previews = PREVIEW_RUNTIME_SCHEMA_VERSIONS.get(schema_id, frozenset())
    return version in previews
