# © Artur Czarnecki. All rights reserved.

"""Bounded consumer inventory for capability-routing TaskEnvelope helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.tenant_x._r1_structural_io import read_production, structural_patterns_present

_ROOT_ADMISSION = "RootExecutionAuthorityAdmissionRequest"
_FROM_ENVELOPE_MARKERS: Final[tuple[str, ...]] = (
    "Task.from_envelope",
    "RuntimeRequest.from_envelope",
)


class CapabilityRoutingConsumerClass(StrEnum):
    CAPABILITY_ROUTING_ONLY = "CAPABILITY_ROUTING_ONLY"
    CANONICAL_EXECUTION_INPUT = "CANONICAL_EXECUTION_INPUT"
    GOVERNANCE_INPUT = "GOVERNANCE_INPUT"
    PROVIDER_OR_STATE_SCOPE = "PROVIDER_OR_STATE_SCOPE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class CapabilityRoutingConsumerRow:
    helper_symbol: str
    consumer_path: str
    classification: CapabilityRoutingConsumerClass
    evidence: tuple[str, ...]


_CAPABILITY_ROUTING_CONSUMER_INVENTORY: Final[tuple[CapabilityRoutingConsumerRow, ...]] = (
    CapabilityRoutingConsumerRow(
        helper_symbol="task_envelope_from_task_context",
        consumer_path="intergrax/runtime/nexus/handoff/coordinator.py",
        classification=CapabilityRoutingConsumerClass.CAPABILITY_ROUTING_ONLY,
        evidence=(
            "resolve_target_agent_id → task_envelope_from_task_context → select_best_routable_agent",
            "envelope used only for RoutableTier2Agent.can_handle scoring",
        ),
    ),
    CapabilityRoutingConsumerRow(
        helper_symbol="task_envelope_for_capability_routing",
        consumer_path="intergrax/runtime/nexus/planning/task_planner.py",
        classification=CapabilityRoutingConsumerClass.CAPABILITY_ROUTING_ONLY,
        evidence=(
            "_sort_agents PRIORITY → task_envelope_for_capability_routing → agent.can_handle(context)",
            "planning agent ordering pre-check — not root execution admission",
        ),
    ),
    CapabilityRoutingConsumerRow(
        helper_symbol="task_envelope_for_capability_routing",
        consumer_path="intergrax/contracts/task_envelope.py",
        classification=CapabilityRoutingConsumerClass.CAPABILITY_ROUTING_ONLY,
        evidence=("definition site — minimal envelope for agent capability pre-check (ACP-CON-6)",),
    ),
    CapabilityRoutingConsumerRow(
        helper_symbol="task_envelope_from_task_context",
        consumer_path="intergrax/runtime/task/agent_capability_intake.py",
        classification=CapabilityRoutingConsumerClass.CAPABILITY_ROUTING_ONLY,
        evidence=("definition site — TaskEnvelope(tenant_id='routing', user_id='routing') sentinel",),
    ),
)


def build_capability_routing_consumer_inventory() -> tuple[CapabilityRoutingConsumerRow, ...]:
    return _CAPABILITY_ROUTING_CONSUMER_INVENTORY


def capability_routing_consumer_counts() -> dict[str, int]:
    inv = build_capability_routing_consumer_inventory()
    counts = {cls.value: 0 for cls in CapabilityRoutingConsumerClass}
    for row in inv:
        counts[row.classification.value] += 1
    counts["capability_routing_helper_consumers_total"] = len(inv)
    return counts


def capability_routing_consumer_unknown_count() -> int:
    return capability_routing_consumer_counts()[CapabilityRoutingConsumerClass.UNKNOWN.value]


def _production_consumer_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                row.consumer_path
                for row in build_capability_routing_consumer_inventory()
                if not row.consumer_path.endswith("agent_capability_intake.py")
                and row.consumer_path != "intergrax/contracts/task_envelope.py"
            }
        )
    )


def routing_sentinel_direct_authority_escape_paths() -> tuple[str, ...]:
    """Direct paths routing sentinel envelope → root admission or canonical from_envelope."""
    escapes: list[str] = []
    intake = read_production("intergrax/runtime/task/agent_capability_intake.py")
    if 'tenant_id="routing"' not in intake and "tenant_id='routing'" not in intake:
        escapes.append("agent_capability_intake: missing routing sentinel literal")

    for rel in _production_consumer_paths():
        text = read_production(rel)
        if _ROOT_ADMISSION in text:
            escapes.append(f"{rel}::{_ROOT_ADMISSION}")
        for marker in _FROM_ENVELOPE_MARKERS:
            if marker in text:
                escapes.append(f"{rel}::{marker}")

    admission_rel = "intergrax/contracts/runtime_execution_admission.py"
    if "routing" in read_production(admission_rel).lower():
        escapes.append(f"{admission_rel}::routing_literal")

    return tuple(escapes)


def assert_capability_routing_sentinel_reconciliation() -> None:
    assert capability_routing_consumer_unknown_count() == 0
    assert routing_sentinel_direct_authority_escape_paths() == ()


def routing_sentinel_semantics_documentation() -> tuple[str, ...]:
    return (
        'tenant_id="routing" is a technical TaskEnvelope shape sentinel for capability pre-check.',
        "It must NOT represent customer, execution, governance, provider, or state tenant identity.",
        "22 P0 mint/rewrite suspects != complete canonical tenant authority inventory.",
        "Canonical root selector remains RootExecutionAuthorityAdmissionRequest outside the P0 suspect set.",
    )


def _structural_sentinel_proof() -> tuple[str, ...]:
    checks: tuple[tuple[str, str], ...] = (
        (
            "intergrax/runtime/task/agent_capability_intake.py",
            'return TaskEnvelope(tenant_id="routing", user_id="routing", metadata=meta)',
        ),
        (
            "intergrax/contracts/task_envelope.py",
            'tenant_id: str = "routing"',
        ),
        (
            "intergrax/runtime/nexus/handoff/coordinator.py",
            "task_envelope_from_task_context(context)",
        ),
        (
            "intergrax/runtime/nexus/handoff/coordinator.py",
            "select_best_routable_agent",
        ),
        (
            "intergrax/runtime/nexus/planning/task_planner.py",
            "task_envelope_for_capability_routing",
        ),
        (
            "intergrax/runtime/nexus/planning/task_planner.py",
            ".can_handle(context)",
        ),
    )
    missing: list[str] = []
    for rel, pattern in checks:
        if pattern not in read_production(rel):
            missing.append(f"{rel}::{pattern}")
    return tuple(missing)


def assert_routing_sentinel_structural_proof() -> None:
    missing = _structural_sentinel_proof()
    assert not missing, f"routing sentinel structural proof gaps: {missing}"
    forbidden = structural_patterns_present(
        "intergrax/runtime/nexus/handoff/coordinator.py",
        ("RootExecutionAuthorityAdmissionRequest", "Task.from_envelope"),
    )
    assert not forbidden
    forbidden_planner = structural_patterns_present(
        "intergrax/runtime/nexus/planning/task_planner.py",
        ("RootExecutionAuthorityAdmissionRequest", "Task.from_envelope", "RuntimeRequest.from_envelope"),
    )
    assert not forbidden_planner
