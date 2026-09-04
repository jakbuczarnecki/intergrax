# © Artur Czarnecki. All rights reserved.

"""Application-owned runtime lifecycle conformance gate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.application_lifecycle_conformance import (
    ApplicationLifecycleConformanceError,
    ApplicationLifecycleRuleId,
    assert_application_lifecycle_conformance,
    collect_application_lifecycle_violations_for_file,
    validate_application_lifecycle_conformance,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[4]


def _write_host_file(root: Path, app_name: str, relative: str, source: str) -> Path:
    path = root / "applications" / app_name / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


def _clean_host_source() -> str:
    return (
        "from intergrax.applications._shared.registry_projection import "
        "MaterializedRegistryProjection\n\n"
        "def create_backend(registry_projection: MaterializedRegistryProjection):\n"
        "    return registry_projection\n"
    )


def test_clean_application_host_passes(tmp_path: Path) -> None:
    _write_host_file(tmp_path, "demo_application", "host/factory.py", _clean_host_source())
    report = assert_application_lifecycle_conformance(tmp_path)
    assert report.ok


def test_direct_agent_registry_construction_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry()\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert any(
        violation.rule_id is ApplicationLifecycleRuleId.AGENT_LIFECYCLE_BYPASS
        for violation in violations
    )


def test_alias_agent_registry_construction_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.runtime.registry.agent_registry import AgentRegistry as AR\n"
        "registry = AR()\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert violations


def test_module_alias_agent_registry_construction_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nimport intergrax.runtime.registry.agent_registry as registry_module\n"
        "registry = registry_module.AgentRegistry()\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert violations


def test_from_agents_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry.from_agents({})\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert any(violation.symbol == "AgentRegistry.from_agents" for violation in violations)


def test_local_register_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry()\n"
        "registry.register(object())\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert any(violation.symbol == "register" for violation in violations)


def test_build_application_registry_in_host_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.applications._shared.wiring import build_application_registry\n"
        "registry = build_application_registry(manifest, context)\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/wiring.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert any(
        violation.rule_id is ApplicationLifecycleRuleId.BUILD_APPLICATION_REGISTRY_BYPASS
        for violation in violations
    )


def test_alias_build_application_registry_fails(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.applications._shared.wiring import "
        "build_application_registry as build_registry\n"
        "registry = build_registry(manifest, context)\n"
    )
    path = _write_host_file(tmp_path, "demo_application", "host/wiring.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        apply_legacy_baseline=False,
    )
    assert violations


def test_agent_registry_read_passes(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.runtime.registry.agent_registry_read import AgentRegistryRead\n"
        "def serve(registry: AgentRegistryRead):\n"
        "    return registry\n"
    )
    _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    report = assert_application_lifecycle_conformance(tmp_path)
    assert report.ok


def test_materialized_registry_projection_passes(tmp_path: Path) -> None:
    _write_host_file(tmp_path, "demo_application", "host/factory.py", _clean_host_source())
    report = assert_application_lifecycle_conformance(tmp_path)
    assert report.ok


def test_unrelated_agent_registry_symbol_passes(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nimport some_other_package as other\n"
        "registry = other.AgentRegistry()\n"
    )
    _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    report = assert_application_lifecycle_conformance(tmp_path)
    assert report.ok


def test_repository_application_host_surface_passes_with_legacy_baseline() -> None:
    report = validate_application_lifecycle_conformance(REPO_ROOT)
    assert report.ok


def test_repository_application_lifecycle_conformance_without_legacy_baseline() -> None:
    report = validate_application_lifecycle_conformance(
        REPO_ROOT,
        apply_legacy_baseline=False,
    )
    assert report.ok


def test_new_violation_in_legacy_file_fails(tmp_path: Path) -> None:
    legacy_path = REPO_ROOT / "applications/lab_application/host/wiring.py"
    source = legacy_path.read_text(encoding="utf-8") + "\nregistry = AgentRegistry()\n"
    path = _write_host_file(tmp_path, "lab_application", "host/wiring.py", source)
    violations = collect_application_lifecycle_violations_for_file(
        path=path,
        repo_root=tmp_path,
        source=source,
        apply_legacy_baseline=True,
    )
    assert any(
        violation.rule_id is ApplicationLifecycleRuleId.AGENT_LIFECYCLE_BYPASS
        for violation in violations
    )


def test_assert_raises_with_formatted_violations(tmp_path: Path) -> None:
    source = (
        _clean_host_source()
        + "\nfrom intergrax.runtime.registry.agent_registry import AgentRegistry\n"
        "registry = AgentRegistry()\n"
    )
    _write_host_file(tmp_path, "demo_application", "host/factory.py", source)
    with pytest.raises(ApplicationLifecycleConformanceError) as exc_info:
        assert_application_lifecycle_conformance(tmp_path)
    assert "APPLICATION_ARCH_AGENT_LIFECYCLE_BYPASS" in str(exc_info.value)
