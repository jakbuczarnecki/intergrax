# © Artur Czarnecki. All rights reserved.

"""CTRL-X semantic-boundary typing SSOT (CX-01..CX-12)."""

from __future__ import annotations

from typing import Final

CTRL_X_R1_START_HEAD: Final[str] = "5e9878d71d363542d25ada19cff2590aa37448d3"

CTRL_X_SEMANTIC_BOUNDARY_MODULES: Final[dict[str, tuple[str, ...]]] = {
    "CX-01": (
        "intergrax/applications/_shared/application_security_wiring.py",
        "intergrax/runtime/security/defense_plugin_loader.py",
        "intergrax/core/security_bootstrap.py",
        "intergrax/applications/_shared/security_wiring.py",
        "intergrax/applications/_shared/security_assembly_resolver.py",
        "intergrax/runtime/security/defense_plugin.py",
        "intergrax/runtime/security/defense_registry.py",
        "intergrax/runtime/security/security_events.py",
        "intergrax/runtime/security/encryption_middleware.py",
    ),
    "CX-02": (
        "intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
        "intergrax/runtime/resilience/dependency_attempt_boundary_composition.py",
    ),
    "CX-03": (
        "intergrax/runtime/execution/budget/models.py",
        "intergrax/applications/_shared/budget_wiring.py",
    ),
    "CX-04": (
        "intergrax/applications/_shared/evaluation_wiring.py",
        "intergrax/runtime/architecture/evaluation_modes.py",
        "intergrax/applications/_shared/evaluation_assembly_resolver.py",
        "intergrax/runtime/architecture/online_evaluation.py",
        "intergrax/runtime/architecture/online_evaluation_registry.py",
        "intergrax/runtime/architecture/online_evaluation_trend.py",
        "intergrax/runtime/architecture/evaluation_registry_trends.py",
        "intergrax/runtime/token_optimization/advisory_evaluation.py",
    ),
    "CX-05": (
        "intergrax/runtime/decision_verification_composition.py",
        "intergrax/applications/_shared/decision_wiring.py",
    ),
    "CX-06": (
        "intergrax/runtime/events/event_bus.py",
        "intergrax/runtime/observability/export_bridge.py",
        "intergrax/runtime/observability/event_delivery/bounded_event_sink.py",
    ),
    "CX-07": (
        "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
        "intergrax/runtime/diagnostics/diagnostic_orchestrator.py",
    ),
    "CX-08": (
        "intergrax/runtime/nexus/tools/runtime_tool_invoker_composition.py",
        "intergrax/runtime/nexus/tools/tool_runtime.py",
        "intergrax/applications/_shared/declarative_tool_wiring.py",
    ),
    "CX-09": (
        "intergrax/skills/resolver.py",
        "intergrax/skills/registry/catalog.py",
    ),
    "CX-10": (
        "intergrax/runtime/registry/agent_registry.py",
        "intergrax/applications/_shared/registry_projection.py",
    ),
    "CX-11": (
        "intergrax/runtime/architecture/capability_graph.py",
        "intergrax/applications/_shared/capability_graph_assembly_resolver.py",
    ),
    "CX-12": (
        "intergrax/context/protocols.py",
        "intergrax/context/assembly_runtime.py",
        "intergrax/context/contracts.py",
    ),
}


def ctrl_x_semantic_boundary_module_paths() -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for plane_id in sorted(CTRL_X_SEMANTIC_BOUNDARY_MODULES):
        for path in CTRL_X_SEMANTIC_BOUNDARY_MODULES[plane_id]:
            if path in seen:
                continue
            seen.add(path)
            ordered.append(path)
    return tuple(ordered)


def ctrl_x_plane_for_boundary_module(module_path: str) -> str | None:
    normalized = module_path.replace("\\", "/")
    for plane_id, paths in CTRL_X_SEMANTIC_BOUNDARY_MODULES.items():
        if normalized in paths:
            return plane_id
    return None
