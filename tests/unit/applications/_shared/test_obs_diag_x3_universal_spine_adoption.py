# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X3 — universal spine adoption, zero-bypass, and operator read backbone gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.obs_diag_surface_qualification import (
    ObsDiagReadExposure,
    ObsDiagSurfaceCoverageError,
    ObsDiagSurfaceKind,
    assert_obs_diag_x3_surface_coverage_complete,
    discover_all_obs_diag_surfaces,
    discover_initialized_scenario_surface_ids,
    iter_obs_diag_surface_descriptor,
)
from intergrax.runtime.architecture.obs_diag_x3_ast_gates import (
    collect_factory_entry_path_violations,
    collect_obs_diag_x3_production_layer_violations,
)
from scripts.proof.scenario_architecture_conformance import (
    assert_all_initialized_scenario_architectures,
    discover_initialized_scenario_slugs,
)
from scripts.maintenance.check_harness_registry_resolution import check_host_wiring_adoption

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_x3_surface_discovery_matches_qualification_registry() -> None:
    assert_obs_diag_x3_surface_coverage_complete(_REPO_ROOT)


def test_x3_new_application_surface_fails_until_qualified(tmp_path: Path) -> None:
    app_dir = tmp_path / "applications" / "unqualified_product_x3"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text("# contract\n", encoding="utf-8")
    (host_dir / "factory.py").write_text(
        "def build():\n    build_harness_host_runtime()\n",
        encoding="utf-8",
    )
    with pytest.raises(ObsDiagSurfaceCoverageError, match="unqualified_product_x3"):
        assert_obs_diag_x3_surface_coverage_complete(tmp_path)


def test_x3_initialized_scenario_discovery_is_dynamic_not_hardcoded() -> None:
    slugs = discover_initialized_scenario_slugs(_REPO_ROOT)
    scenario_surfaces = discover_initialized_scenario_surface_ids(_REPO_ROOT)
    assert scenario_surfaces == frozenset(f"scenario:{slug}" for slug in slugs)
    assert len(slugs) >= 1


def test_x3_all_initialized_scenarios_pass_architecture_conformance() -> None:
    assert_all_initialized_scenario_architectures(_REPO_ROOT)


def test_x3_application_factory_and_host_wiring_canonical() -> None:
    assert collect_factory_entry_path_violations(_REPO_ROOT) == []
    assert check_host_wiring_adoption(repo_root=_REPO_ROOT) == []


def test_x3_product_application_factories_use_harness_runtime_marker() -> None:
    discovered_apps = discover_all_obs_diag_surfaces(_REPO_ROOT)
    for app_id in sorted(discovered_apps):
        if app_id.startswith(("scenario:", "worker:", "harness:")):
            continue
        descriptor = iter_obs_diag_surface_descriptor(app_id)
        if descriptor.kind is ObsDiagSurfaceKind.LAB_APPLICATION:
            continue
        factory = _REPO_ROOT / "applications" / app_id / "host" / "factory.py"
        composition = (
            _REPO_ROOT
            / "applications"
            / app_id
            / "host"
            / "host_runtime_composition.py"
        )
        if factory.is_file():
            text = factory.read_text(encoding="utf-8")
            if "build_harness_host_runtime" in text:
                continue
        if composition.is_file():
            text = composition.read_text(encoding="utf-8")
            assert "build_harness_host_runtime" in text, app_id
            continue
        if factory.is_file():
            pytest.fail(f"{app_id}: missing canonical harness host runtime entry")


def test_x3_no_illegal_local_diagnostic_authority_in_production_layers() -> None:
    violations = collect_obs_diag_x3_production_layer_violations(_REPO_ROOT)
    assert violations == []


def test_x3_native_read_product_host_wires_diagnostic_read_service() -> None:
    wiring_path = (
        _REPO_ROOT
        / "intergrax"
        / "applications"
        / "_shared"
        / "product_observability_dashboard_wiring.py"
    )
    source = wiring_path.read_text(encoding="utf-8")
    assert "DiagnosticReadService" in source
    assert "resolve_host_diagnostic_read_service" in source
    descriptor = iter_obs_diag_surface_descriptor("governed_contractor_application")
    assert descriptor.read_exposure is ObsDiagReadExposure.NATIVE


@pytest.mark.parametrize(
    "scenario_slug",
    discover_initialized_scenario_slugs(_REPO_ROOT),
)
def test_x3_qualified_scenario_surfaces_registered(scenario_slug: str) -> None:
    surface_id = f"scenario:{scenario_slug}"
    descriptor = iter_obs_diag_surface_descriptor(surface_id)
    assert descriptor.kind is ObsDiagSurfaceKind.SCENARIO
