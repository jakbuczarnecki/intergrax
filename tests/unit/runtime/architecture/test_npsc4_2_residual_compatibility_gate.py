# © Artur Czarnecki. All rights reserved.

"""NPSC-4.2 — residual compatibility and legacy seam retirement gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_APPLICATIONS_ROOT = _REPO_ROOT / "applications"
_SHARED_ROOT = _REPO_ROOT / "intergrax" / "applications" / "_shared"
_COMPOSITION_PATH = _SHARED_ROOT / "harness_host_composition.py"
_RUNTIME_PATH = _SHARED_ROOT / "harness_host_runtime.py"

_RETIRED_TOKENS = (
    "resolve_harness_host_nexus_loop_legacy",
    "HarnessHostLegacyComposition",
    "harness_host_runtime_compat",
    "_legacy_composition",
)

_TIER3_HOST_ROOTS = tuple(
    path
    for path in _APPLICATIONS_ROOT.glob("*/host")
    if path.is_dir() and "docker" not in path.parts
)

_ALLOWED_RAW_NEXUS_IMPORTERS = frozenset(
    {
        "intergrax/applications/_shared/harness_host_composition.py",
        "intergrax/applications/_shared/harness_host_runtime.py",
        "intergrax/applications/_shared/nexus_factory.py",
        "intergrax/applications/_shared/host_task_execution_wiring.py",
        "intergrax/applications/_shared/platform_wiring.py",
        "intergrax/applications/_shared/plugin_bootstrap.py",
        "intergrax/applications/_shared/security_wiring.py",
        "intergrax/applications/_shared/application_security_wiring.py",
        "intergrax/applications/_shared/application_host_wiring.py",
        "intergrax/applications/_shared/guardrail_wiring.py",
        "intergrax/applications/_shared/reliability_wiring.py",
        "intergrax/applications/_shared/decision_wiring.py",
        "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
        "intergrax/applications/_shared/environment_snapshot_wiring.py",
        "intergrax/applications/_shared/capability_alias_intake_wiring.py",
        "intergrax/applications/_shared/guardrail_assembly_resolver.py",
        "intergrax/applications/_shared/security_assembly_resolver.py",
        "intergrax/applications/_shared/scenario_runtime_baseline.py",
        "intergrax/debug/app.py",
        "intergrax/harness/lab_fastapi.py",
        "scripts/maintenance/check_harness_security_wiring.py",
    }
)


def _relative(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _python_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
        and "docker" not in path.parts
        and "runtime-context" not in path.parts
    ]


def _source_contains_tokens(path: Path, tokens: tuple[str, ...]) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    rel = _relative(path)
    return [f"{rel}: forbidden token {token!r}" for token in tokens if token in source]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.gate
def test_npsc42_retired_legacy_compat_tokens_absent_from_production() -> None:
    violations: list[str] = []
    scan_roots = (
        _REPO_ROOT / "intergrax",
        _REPO_ROOT / "applications",
        _REPO_ROOT / "scripts",
        _REPO_ROOT / "proof_infrastructure",
    )
    for root in scan_roots:
        for path in _python_files(root):
            if path.name.startswith("test_") or "/tests/" in _relative(path):
                continue
            violations.extend(_source_contains_tokens(path, _RETIRED_TOKENS))
    assert violations == [], "retired legacy compat tokens found:\n" + "\n".join(violations)


@pytest.mark.gate
@pytest.mark.parametrize("host_root", _TIER3_HOST_ROOTS, ids=lambda p: p.parent.name)
def test_npsc42_tier3_host_factories_have_no_legacy_compat_imports(host_root: Path) -> None:
    violations: list[str] = []
    for path in host_root.rglob("*.py"):
        if "docker" in path.parts or "runtime-context" in path.parts:
            continue
        violations.extend(_source_contains_tokens(path, _RETIRED_TOKENS))
        modules = _imported_modules(path)
        if "intergrax.applications._shared.harness_host_runtime_compat" in modules:
            violations.append(f"{_relative(path)}: imports harness_host_runtime_compat")
    assert violations == [], "\n".join(violations)


@pytest.mark.gate
def test_npsc42_harness_host_composition_exposes_narrow_capabilities() -> None:
    source = _COMPOSITION_PATH.read_text(encoding="utf-8-sig")
    assert "class HarnessHostInternalComposition" in source
    assert "resolve_harness_host_execution_terminal" in source
    assert "resolve_harness_host_nexus_loop_legacy" not in source
    assert "HarnessHostLegacyComposition" not in source


@pytest.mark.gate
def test_npsc42_harness_host_runtime_uses_internal_composition_field() -> None:
    source = _RUNTIME_PATH.read_text(encoding="utf-8-sig")
    assert "_internal_composition: HarnessHostInternalComposition" in source
    assert "_legacy_composition" not in source


@pytest.mark.gate
def test_npsc42_raw_nexus_imports_remain_composition_owner_allowlist() -> None:
    violations: list[str] = []
    for path in _python_files(_SHARED_ROOT):
        rel = _relative(path)
        if rel in _ALLOWED_RAW_NEXUS_IMPORTERS:
            continue
        if "from intergrax.runtime.nexus.nexus_loop import NexusLoop" in path.read_text(encoding="utf-8-sig"):
            violations.append(rel)
    assert violations == [], (
        "unexpected NexusLoop imports in shared application wiring:\n" + "\n".join(violations)
    )


_ALLOWED_ORCHESTRATION_BACKEND_ACCESSORS = frozenset(
    {
        "intergrax/applications/_shared/harness_host_composition.py",
        "applications/local_workspace_application/model_runtime_proof/runtime.py",
        "scripts/maintenance/check_harness_security_wiring.py",
        "scripts/maintenance/check_harness_reliability_wiring.py",
    }
)


def _source_references_orchestration_backend(path: Path) -> bool:
    return "_orchestration_backend" in path.read_text(encoding="utf-8-sig")


@pytest.mark.gate
def test_npsc42_orchestration_backend_access_confined_to_allowlist() -> None:
    violations: list[str] = []
    scan_roots = (
        _REPO_ROOT / "intergrax",
        _REPO_ROOT / "applications",
        _REPO_ROOT / "scripts",
        _REPO_ROOT / "proof_infrastructure",
    )
    for root in scan_roots:
        for path in _python_files(root):
            if path.name.startswith("test_") or "/tests/" in _relative(path):
                continue
            if not _source_references_orchestration_backend(path):
                continue
            rel = _relative(path)
            if rel not in _ALLOWED_ORCHESTRATION_BACKEND_ACCESSORS:
                violations.append(rel)
    assert violations == [], (
        "unexpected production _orchestration_backend access:\n" + "\n".join(violations)
    )


@pytest.mark.gate
def test_npsc42_harness_host_platform_bootstrap_has_no_trace_type_suppression() -> None:
    source = _COMPOSITION_PATH.read_text(encoding="utf-8-sig")
    assert "bootstrap_harness_host_platform" in source
    assert "# type: ignore[arg-type]" not in source
    assert "cast(" not in source
