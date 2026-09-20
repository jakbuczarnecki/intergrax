# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X3 — universal spine adoption, zero-bypass, and operator read backbone gates."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.execution_surface_discovery import (
    discover_worker_execution_surfaces,
    iter_worker_execution_surface_python_paths,
)
from intergrax.applications._shared.obs_diag_surface_qualification import (
    ObsDiagReadExposure,
    ObsDiagSurfaceClassificationError,
    ObsDiagSurfaceCoverageError,
    ObsDiagSurfaceKind,
    ObsDiagSurfaceRegistryIntegrityError,
    assert_obs_diag_x3_surface_coverage_complete,
    discover_all_obs_diag_surfaces,
    discover_initialized_scenario_surface_ids,
    discover_worker_surface_ids,
    iter_obs_diag_surface_descriptor,
    validate_obs_diag_surface_registry_integrity,
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


def test_x3a_worker_discovery_uses_bootstrap_surface_marker_not_filename() -> None:
    surfaces = discover_worker_execution_surfaces(_REPO_ROOT)
    surface_ids = {surface.surface_id for surface in surfaces}
    assert "worker:local_workspace_application.background" in surface_ids
    assert all(
        path.is_file()
        for surface in surfaces
        for path in surface.entry_paths
    )


def test_x3a_ast_worker_scan_matches_discovery_paths() -> None:
    discovered_paths = {
        path.resolve()
        for path in iter_worker_execution_surface_python_paths(_REPO_ROOT)
    }
    from intergrax.runtime.architecture.obs_diag_x3_ast_gates import _iter_worker_entry_python

    ast_paths = {path.resolve() for path in _iter_worker_entry_python(_REPO_ROOT)}
    assert ast_paths == discovered_paths


def test_x3a_new_worker_outside_local_workspace_fails_until_qualified(tmp_path: Path) -> None:
    app_dir = tmp_path / "applications" / "example_product_x3a"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text(
        "from intergrax.applications.contracts.manifest import ApplicationManifest\n"
        "MANIFEST = ApplicationManifest.product(\n"
        '    app_id="example_product_x3a",\n'
        '    name="Example",\n'
        '    route_prefix="/v1/example",\n'
        '    env_prefix="EXAMPLE_",\n'
        ")\n",
        encoding="utf-8",
    )
    (host_dir / "queue_consumer_main.py").write_text(
        '_QUEUE_PROCESS_ROLE = "background_worker"\n'
        "def main():\n"
        "    BootstrapSurfaceKind.WORKER_BACKGROUND\n",
        encoding="utf-8",
    )
    worker_ids = discover_worker_surface_ids(tmp_path)
    assert "worker:example_product_x3a.background" in worker_ids
    with pytest.raises(ObsDiagSurfaceCoverageError, match="worker:example_product_x3a.background"):
        assert_obs_diag_x3_surface_coverage_complete(tmp_path)


def test_x3a_lab_to_product_manifest_drift_fails(tmp_path: Path) -> None:
    app_dir = tmp_path / "applications" / "drift_lab_to_product_x3a"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text(
        "from intergrax.applications.contracts.manifest import ApplicationManifest\n"
        "MANIFEST = ApplicationManifest.product(\n"
        '    app_id="drift_lab_to_product_x3a",\n'
        '    name="Drift",\n'
        '    route_prefix="/v1/drift",\n'
        '    env_prefix="DRIFT_",\n'
        ")\n",
        encoding="utf-8",
    )
    (host_dir / "factory.py").write_text("# host\n", encoding="utf-8")
    import intergrax.applications._shared.obs_diag_surface_qualification as qualification

    original = qualification.OBS_DIAG_X3_QUALIFIED_SURFACES
    qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = (
        qualification.ObsDiagSurfaceDescriptor(
            surface_id="drift_lab_to_product_x3a",
            kind=qualification.ObsDiagSurfaceKind.LAB_APPLICATION,
            production_capable=False,
            entry_mechanism="test",
            read_exposure=qualification.ObsDiagReadExposure.NOT_APPLICABLE,
            exemption=qualification.ObsDiagSurfaceExemption.LAB,
            exemption_reason="stale lab qualification",
        ),
    )
    try:
        with pytest.raises(ObsDiagSurfaceClassificationError, match="drift_lab_to_product_x3a"):
            assert_obs_diag_x3_surface_coverage_complete(tmp_path)
    finally:
        qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = original


def test_x3a_product_to_lab_manifest_drift_fails(tmp_path: Path) -> None:
    app_dir = tmp_path / "applications" / "drift_product_to_lab_x3a"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text(
        "from intergrax.applications.contracts.manifest import ApplicationManifest\n"
        "MANIFEST = ApplicationManifest.lab(\n"
        '    app_id="drift_product_to_lab_x3a",\n'
        '    name="Drift",\n'
        '    route_prefix="/v1/drift",\n'
        '    env_prefix="DRIFT_",\n'
        ")\n",
        encoding="utf-8",
    )
    (host_dir / "factory.py").write_text("# host\n", encoding="utf-8")
    import intergrax.applications._shared.obs_diag_surface_qualification as qualification

    original = qualification.OBS_DIAG_X3_QUALIFIED_SURFACES
    qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = (
        qualification.ObsDiagSurfaceDescriptor(
            surface_id="drift_product_to_lab_x3a",
            kind=qualification.ObsDiagSurfaceKind.PRODUCT_APPLICATION,
            production_capable=True,
            entry_mechanism="test",
            read_exposure=qualification.ObsDiagReadExposure.OPTIONAL,
        ),
    )
    try:
        with pytest.raises(ObsDiagSurfaceClassificationError, match="drift_product_to_lab_x3a"):
            assert_obs_diag_x3_surface_coverage_complete(tmp_path)
    finally:
        qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = original


def test_x3a_new_worker_illegal_nexus_bypass_ast_gate_fails(tmp_path: Path) -> None:
    app_dir = tmp_path / "applications" / "illegal_worker_x3a"
    host_dir = app_dir / "host"
    host_dir.mkdir(parents=True)
    (app_dir / "manifest.py").write_text("# contract\n", encoding="utf-8")
    (host_dir / "rogue_worker_main.py").write_text(
        '_EVIL_PROCESS_ROLE = "background_worker"\n'
        "def main():\n"
        "    NexusLoop()\n"
        "    BootstrapSurfaceKind.WORKER_BACKGROUND\n",
        encoding="utf-8",
    )
    violations = collect_obs_diag_x3_production_layer_violations(tmp_path)
    assert any(
        violation.symbol == "NexusLoop" and "rogue_worker_main.py" in violation.relative_path
        for violation in violations
    )


def test_x3a_registry_rejects_production_capable_with_lab_exemption() -> None:
    import intergrax.applications._shared.obs_diag_surface_qualification as qualification

    original = qualification.OBS_DIAG_X3_QUALIFIED_SURFACES
    qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = original + (
        qualification.ObsDiagSurfaceDescriptor(
            surface_id="invalid_exemption_x3a",
            kind=qualification.ObsDiagSurfaceKind.WORKER,
            production_capable=True,
            entry_mechanism="test",
            read_exposure=qualification.ObsDiagReadExposure.NOT_APPLICABLE,
            exemption=qualification.ObsDiagSurfaceExemption.LAB,
            exemption_reason="invalid",
        ),
    )
    try:
        with pytest.raises(ObsDiagSurfaceRegistryIntegrityError, match="invalid_exemption_x3a"):
            validate_obs_diag_surface_registry_integrity()
    finally:
        qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = original


def test_x3a_duplicate_surface_id_in_registry_fails() -> None:
    import intergrax.applications._shared.obs_diag_surface_qualification as qualification

    duplicate = qualification.OBS_DIAG_X3_QUALIFIED_SURFACES[0]
    original = qualification.OBS_DIAG_X3_QUALIFIED_SURFACES
    qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = original + (duplicate,)
    try:
        with pytest.raises(ObsDiagSurfaceRegistryIntegrityError, match="duplicate"):
            validate_obs_diag_surface_registry_integrity()
    finally:
        qualification.OBS_DIAG_X3_QUALIFIED_SURFACES = original
