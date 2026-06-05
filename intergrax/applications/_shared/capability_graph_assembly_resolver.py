# © Artur Czarnecki. All rights reserved.

"""Capability graph assembly validation for Tier-3 hosts (Phase CG-2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from intergrax.applications._shared.capability_graph_wiring import EnvironmentCapabilityGraphView
from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.architecture.capability_graph_applications import application_capability_node_id


@dataclass(frozen=True, slots=True)
class CapabilityGraphAssemblyValidationResult:
    """Outcome of environment capability graph assembly validation."""

    valid: bool
    errors: tuple[str, ...] = ()


class CapabilityGraphAssemblyError(ValueError):
    """Raised when environment capability graph assembly validation fails."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors: tuple[str, ...] = tuple(errors)
        message = "; ".join(self.errors)
        super().__init__(message)


def validate_environment_capability_graph(
    view: EnvironmentCapabilityGraphView,
    snapshot: HarnessRegistrySnapshot,
    manifest: ApplicationManifest,
) -> CapabilityGraphAssemblyValidationResult:
    """Validate wired registry artifacts appear as typed nodes in the environment graph."""
    errors: list[str] = []
    application_node = application_capability_node_id(manifest)

    if not view.contains_node(application_node):
        errors.append(f"missing application node {application_node!r} in capability graph")

    for tool_id in snapshot.tool_ids():
        node_id = f"tool:{tool_id}"
        if not view.contains_node(node_id):
            errors.append(f"missing catalog tool node {node_id!r} in capability graph")

    for skill_id in snapshot.skill_ids():
        node_id = f"skill:{skill_id}"
        if not view.contains_node(node_id):
            errors.append(f"missing catalog skill node {node_id!r} in capability graph")

    enabled_agents = [binding for binding in manifest.enabled_agents()]
    if enabled_agents:
        agent_nodes = [node_id for node_id in view.node_ids() if node_id.startswith("agent:")]
        if not agent_nodes:
            errors.append("capability graph must include at least one agent node for enabled roster")

    return CapabilityGraphAssemblyValidationResult(valid=not errors, errors=tuple(errors))


def assert_capability_graph_assembly_valid(
    view: EnvironmentCapabilityGraphView,
    snapshot: HarnessRegistrySnapshot,
    manifest: ApplicationManifest,
) -> None:
    """Raise :class:`CapabilityGraphAssemblyError` when capability graph validation fails."""
    result = validate_environment_capability_graph(view, snapshot, manifest)
    if not result.valid:
        raise CapabilityGraphAssemblyError(result.errors)
