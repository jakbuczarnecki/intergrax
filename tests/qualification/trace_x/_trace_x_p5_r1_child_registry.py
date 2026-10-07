# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1-R1-Q1: independent ChildExecutionRunner surface classification registry."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Final

from tests.qualification.trace_x._trace_x_p4_registry_types import (
    SurfaceParityResult,
    compare_discovered_to_registry,
)


class ChildExecutionRunnerSurfaceClassification(enum.StrEnum):
    PROFILE_AWARE_CAPABLE_PRODUCTION = "PROFILE_AWARE_CAPABLE_PRODUCTION"
    GENERIC_PRODUCTION = "GENERIC_PRODUCTION"
    INTERNAL_SANCTIONED = "INTERNAL_SANCTIONED"
    NON_PRODUCTION = "NON_PRODUCTION"


@dataclass(frozen=True, slots=True)
class RegisteredChildExecutionRunnerSurface:
    surface_key: str
    classification: ChildExecutionRunnerSurfaceClassification
    profile_semantics: str
    expected_inheritance_behavior: str
    owner: str
    rationale: str

    @property
    def key(self) -> tuple[str, str]:
        path, enclosing = self.surface_key.split("::", 1)
        return (path, enclosing)


CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY: Final[tuple[RegisteredChildExecutionRunnerSurface, ...]] = (
    RegisteredChildExecutionRunnerSurface(
        surface_key="intergrax/runtime/nexus/execution/graph_executor.py::GraphExecutor.__init__",
        classification=ChildExecutionRunnerSurfaceClassification.PROFILE_AWARE_CAPABLE_PRODUCTION,
        profile_semantics="Profile-aware when host composition supplies ChildExecutionContextInheritancePort",
        expected_inheritance_behavior="Forwards child_context_inheritance keyword to internal ChildExecutionRunner",
        owner="Nexus / GraphExecutor",
        rationale="Canonical Nexus graph child execution; host orchestration may inject profile pinning seam",
    ),
    RegisteredChildExecutionRunnerSurface(
        surface_key="intergrax/runtime/execution/execution_work_port.py::ChildExecutionWorkPort.__init__",
        classification=ChildExecutionRunnerSurfaceClassification.GENERIC_PRODUCTION,
        profile_semantics="Profile-capable by explicit injection at composition root",
        expected_inheritance_behavior="child_context_inheritance may be None for generic execution roots",
        owner="Execution / work port composition",
        rationale="Generic child work port; does not establish effective-profile host context",
    ),
    RegisteredChildExecutionRunnerSurface(
        surface_key=(
            "intergrax/runtime/execution/execution_work_port.py::DelegatedSubtaskChildExecutionWorkPort.__init__"
        ),
        classification=ChildExecutionRunnerSurfaceClassification.GENERIC_PRODUCTION,
        profile_semantics="Profile-capable by explicit injection at composition root",
        expected_inheritance_behavior="child_context_inheritance may be None for delegated-subtask generic roots",
        owner="Execution / delegated subtask work port",
        rationale="U4 delegated specialist child path without mandatory host effective-profile ownership",
    ),
    RegisteredChildExecutionRunnerSurface(
        surface_key=(
            "intergrax/runtime/execution/execution_work_port.py::DelegatedProviderChildExecutionEngine.__init__"
        ),
        classification=ChildExecutionRunnerSurfaceClassification.GENERIC_PRODUCTION,
        profile_semantics="Profile-capable by explicit injection at composition root",
        expected_inheritance_behavior="child_context_inheritance may be None for provider delegation roots",
        owner="Execution / delegated provider engine",
        rationale="P2.1 provider adoption child owner; does not establish effective-profile host context",
    ),
)


@dataclass(frozen=True, slots=True)
class RegisteredProfileAwareWireHostRoot:
    surface_key: str
    owner: str
    rationale: str

    @property
    def key(self) -> tuple[str, str]:
        path, enclosing = self.surface_key.split("::", 1)
        return (path, enclosing)


PROFILE_AWARE_WIRE_HOST_ROOT_REGISTRY: Final[tuple[RegisteredProfileAwareWireHostRoot, ...]] = (
    RegisteredProfileAwareWireHostRoot(
        surface_key=(
            "intergrax/applications/_shared/scenario_runtime_baseline.py::build_scenario_runtime_from_environment"
        ),
        owner="Applications / scenario runtime baseline",
        rationale="Canonical scenario host establishes effective-profile execution via wire_host_effective_profile_execution",
    ),
)


@dataclass(frozen=True, slots=True)
class GenericCompositionDriftWatch:
    module_path: str
    rationale: str
    forbidden_profile_aware_markers: tuple[str, ...] = (
        "wire_host_effective_profile_execution",
        "HostEffectiveProfileExecutionWiring",
    )


GENERIC_COMPOSITION_DRIFT_WATCH_REGISTRY: Final[tuple[GenericCompositionDriftWatch, ...]] = (
    GenericCompositionDriftWatch(
        module_path="intergrax/applications/_shared/production_agent_capability_runtime.py",
        rationale=(
            "ProductionAgentCapabilityRuntime does not own effective-profile host execution; "
            "delegated-subtask child port remains generic unless explicitly reclassified"
        ),
    ),
    GenericCompositionDriftWatch(
        module_path="intergrax/applications/_shared/production_delegated_subtask_child_execution_wiring.py",
        rationale="Process-level delegated subtask binding; optional inheritance injection only",
    ),
)


def _registry_keys(registry: tuple[RegisteredChildExecutionRunnerSurface, ...]) -> frozenset[tuple[str, str]]:
    return frozenset(row.key for row in registry)


def compare_child_runner_surfaces_to_registry(
    discovered: frozenset[str],
) -> SurfaceParityResult:
    discovered_keys = frozenset(
        (surface_key.split("::", 1)[0], surface_key.split("::", 1)[1]) for surface_key in discovered
    )
    return compare_discovered_to_registry(discovered_keys, CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY)


def compare_profile_aware_wire_host_roots_to_registry(
    discovered: frozenset[str],
) -> SurfaceParityResult:
    discovered_keys = frozenset(
        (surface_key.split("::", 1)[0], surface_key.split("::", 1)[1]) for surface_key in discovered
    )
    return compare_discovered_to_registry(discovered_keys, PROFILE_AWARE_WIRE_HOST_ROOT_REGISTRY)


def registry_classification_ambiguity_violations() -> list[str]:
    violations: list[str] = []
    for row in CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY:
        if not row.classification:
            violations.append(f"empty classification: {row.surface_key}")
        if not row.profile_semantics.strip():
            violations.append(f"empty profile_semantics: {row.surface_key}")
        if row.classification == ChildExecutionRunnerSurfaceClassification.GENERIC_PRODUCTION:
            rationale_lower = row.rationale.lower()
            if (
                "generic" not in rationale_lower
                and "does not own" not in rationale_lower
                and "does not establish" not in rationale_lower
                and "without mandatory host" not in rationale_lower
            ):
                violations.append(f"generic surface missing architectural rationale: {row.surface_key}")
    return violations


__all__ = [
    "CHILD_EXECUTION_RUNNER_SURFACE_REGISTRY",
    "ChildExecutionRunnerSurfaceClassification",
    "GENERIC_COMPOSITION_DRIFT_WATCH_REGISTRY",
    "PROFILE_AWARE_WIRE_HOST_ROOT_REGISTRY",
    "RegisteredChildExecutionRunnerSurface",
    "compare_child_runner_surfaces_to_registry",
    "compare_profile_aware_wire_host_roots_to_registry",
    "registry_classification_ambiguity_violations",
]
