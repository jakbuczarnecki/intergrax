# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X3 — typed production surface inventory and anti-drift qualification registry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from intergrax.applications._shared.application_runtime_graph import list_application_projects
from scripts.proof.scenario_architecture_conformance import discover_initialized_scenario_slugs


class ObsDiagSurfaceKind(StrEnum):
    PRODUCT_APPLICATION = "product_application"
    LAB_APPLICATION = "lab_application"
    WORKER = "worker"
    SCENARIO = "scenario"
    HARNESS_ENTRY = "harness_entry"


class ObsDiagReadExposure(StrEnum):
    NATIVE = "NATIVE"
    OPTIONAL = "OPTIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    MISSING = "MISSING"


class ObsDiagSurfaceExemption(StrEnum):
    LAB = "LAB"
    DEBUG = "DEBUG"
    TESTING = "TESTING"
    DESIGN_ONLY = "DESIGN_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class ObsDiagSurfaceDescriptor:
    """One qualified OBS/DIAG adoption surface (write spine + read expectation)."""

    surface_id: str
    kind: ObsDiagSurfaceKind
    production_capable: bool
    entry_mechanism: str
    read_exposure: ObsDiagReadExposure
    exemption: ObsDiagSurfaceExemption | None = None
    exemption_reason: str | None = None


# Every discovered production-relevant surface must appear here (anti-drift).
OBS_DIAG_X3_QUALIFIED_SURFACES: tuple[ObsDiagSurfaceDescriptor, ...] = (
    ObsDiagSurfaceDescriptor(
        surface_id="governed_contractor_application",
        kind=ObsDiagSurfaceKind.PRODUCT_APPLICATION,
        production_capable=True,
        entry_mechanism="HTTP → HostTaskExecution → UnifiedTaskRunner → execute_root_task",
        read_exposure=ObsDiagReadExposure.NATIVE,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="legal_application",
        kind=ObsDiagSurfaceKind.PRODUCT_APPLICATION,
        production_capable=True,
        entry_mechanism="HTTP → build_harness_environment_host_task_execution → UnifiedTaskRunner",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="dispute_sim_application",
        kind=ObsDiagSurfaceKind.PRODUCT_APPLICATION,
        production_capable=True,
        entry_mechanism="HTTP → harness host task execution → UnifiedTaskRunner",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="local_workspace_application",
        kind=ObsDiagSurfaceKind.PRODUCT_APPLICATION,
        production_capable=True,
        entry_mechanism="HTTP + background worker → build_harness_host_runtime spine",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="research_application",
        kind=ObsDiagSurfaceKind.PRODUCT_APPLICATION,
        production_capable=True,
        entry_mechanism="HTTP → harness host task execution (DEV api environment)",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="lab_application",
        kind=ObsDiagSurfaceKind.LAB_APPLICATION,
        production_capable=False,
        entry_mechanism="build_harness_host_runtime (lab profile)",
        read_exposure=ObsDiagReadExposure.NOT_APPLICABLE,
        exemption=ObsDiagSurfaceExemption.LAB,
        exemption_reason="Explicit lab host; diagnostics posture profile-driven",
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="poc_template_application",
        kind=ObsDiagSurfaceKind.LAB_APPLICATION,
        production_capable=False,
        entry_mechanism="build_harness_host_runtime scaffold",
        read_exposure=ObsDiagReadExposure.NOT_APPLICABLE,
        exemption=ObsDiagSurfaceExemption.LAB,
        exemption_reason="Scaffold only",
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="attestation_demo",
        kind=ObsDiagSurfaceKind.LAB_APPLICATION,
        production_capable=False,
        entry_mechanism="build_harness_host_runtime partner PoC",
        read_exposure=ObsDiagReadExposure.NOT_APPLICABLE,
        exemption=ObsDiagSurfaceExemption.LAB,
        exemption_reason="Partner PoC",
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="intergrax_assistant_application",
        kind=ObsDiagSurfaceKind.LAB_APPLICATION,
        production_capable=False,
        entry_mechanism="build_harness_host_runtime scaffold",
        read_exposure=ObsDiagReadExposure.NOT_APPLICABLE,
        exemption=ObsDiagSurfaceExemption.LAB,
        exemption_reason="Lab scaffold",
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="worker:local_workspace_application.background",
        kind=ObsDiagSurfaceKind.WORKER,
        production_capable=True,
        entry_mechanism="background_worker_main → build_harness_host_runtime → queue execute",
        read_exposure=ObsDiagReadExposure.NOT_APPLICABLE,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="harness:intergrax.harness.app",
        kind=ObsDiagSurfaceKind.HARNESS_ENTRY,
        production_capable=False,
        entry_mechanism="CLI harness → build_harness_host_runtime",
        read_exposure=ObsDiagReadExposure.NOT_APPLICABLE,
        exemption=ObsDiagSurfaceExemption.DEBUG,
        exemption_reason="Developer harness entry",
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="scenario:ai_incident_investigation",
        kind=ObsDiagSurfaceKind.SCENARIO,
        production_capable=True,
        entry_mechanism="execute_scenario_task → UnifiedTaskRunner",
        read_exposure=ObsDiagReadExposure.NATIVE,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="scenario:enterprise_payment_uncertainty_recovery",
        kind=ObsDiagSurfaceKind.SCENARIO,
        production_capable=True,
        entry_mechanism="ScenarioRuntimeBaseline + lab ERL spine",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="scenario:indirect_prompt_injection",
        kind=ObsDiagSurfaceKind.SCENARIO,
        production_capable=True,
        entry_mechanism="execute_scenario_task → platform proof harness",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
    ObsDiagSurfaceDescriptor(
        surface_id="scenario:verified_product_identification",
        kind=ObsDiagSurfaceKind.SCENARIO,
        production_capable=True,
        entry_mechanism="execute_scenario_task → UnifiedTaskRunner",
        read_exposure=ObsDiagReadExposure.OPTIONAL,
    ),
)


class ObsDiagSurfaceCoverageError(Exception):
    """Raised when discovered surfaces are not fully qualified in the X3 registry."""


def _qualified_surface_ids() -> frozenset[str]:
    return frozenset(descriptor.surface_id for descriptor in OBS_DIAG_X3_QUALIFIED_SURFACES)


def discover_application_surface_ids(repo_root: Path) -> frozenset[str]:
    return frozenset(list_application_projects(repo_root))


def discover_initialized_scenario_surface_ids(repo_root: Path) -> frozenset[str]:
    return frozenset(
        f"scenario:{slug}" for slug in discover_initialized_scenario_slugs(repo_root)
    )


def discover_worker_surface_ids(repo_root: Path) -> frozenset[str]:
    surfaces: set[str] = set()
    worker_factory = (
        repo_root
        / "applications"
        / "local_workspace_application"
        / "host"
        / "background_worker_factory.py"
    )
    if worker_factory.is_file():
        surfaces.add("worker:local_workspace_application.background")
    return frozenset(surfaces)


def discover_harness_entry_surface_ids(repo_root: Path) -> frozenset[str]:
    harness_app = repo_root / "intergrax" / "harness" / "app.py"
    if harness_app.is_file():
        return frozenset({"harness:intergrax.harness.app"})
    return frozenset()


def discover_all_obs_diag_surfaces(repo_root: Path) -> frozenset[str]:
    discovered = set(discover_application_surface_ids(repo_root))
    discovered.update(discover_initialized_scenario_surface_ids(repo_root))
    discovered.update(discover_worker_surface_ids(repo_root))
    discovered.update(discover_harness_entry_surface_ids(repo_root))
    return frozenset(discovered)


def assert_obs_diag_x3_surface_coverage_complete(repo_root: Path) -> None:
    discovered = discover_all_obs_diag_surfaces(repo_root)
    qualified = _qualified_surface_ids()
    missing = sorted(discovered - qualified)
    stale = sorted(qualified - discovered)
    if missing or stale:
        parts: list[str] = []
        if missing:
            parts.append(f"unqualified surfaces: {missing}")
        if stale:
            parts.append(f"stale qualification entries (not discovered): {stale}")
        raise ObsDiagSurfaceCoverageError("; ".join(parts))


def iter_obs_diag_surface_descriptor(surface_id: str) -> ObsDiagSurfaceDescriptor:
    for descriptor in OBS_DIAG_X3_QUALIFIED_SURFACES:
        if descriptor.surface_id == surface_id:
            return descriptor
    raise KeyError(surface_id)


__all__ = [
    "OBS_DIAG_X3_QUALIFIED_SURFACES",
    "ObsDiagReadExposure",
    "ObsDiagSurfaceCoverageError",
    "ObsDiagSurfaceDescriptor",
    "ObsDiagSurfaceExemption",
    "ObsDiagSurfaceKind",
    "assert_obs_diag_x3_surface_coverage_complete",
    "discover_all_obs_diag_surfaces",
    "discover_application_surface_ids",
    "discover_harness_entry_surface_ids",
    "discover_initialized_scenario_surface_ids",
    "discover_worker_surface_ids",
    "iter_obs_diag_surface_descriptor",
]
