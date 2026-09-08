# © Artur Czarnecki. All rights reserved.

"""Stage 12 isolation assessment traceability and no-runtime-change gates."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.environment_wiring import wire_application_environment
from intergrax.core.plugins.platform_qualification import PlatformPluginTrustModel

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ADR_PATH = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "technical"
    / "adr"
    / "entries"
    / "2026-09-07"
    / "ADR-SEC-002.md"
)
_PLATFORM_PLUGINS_DOC = _REPO_ROOT / "docs" / "project" / "architecture" / "PLATFORM_PLUGINS.md"
_CATALOG_DOC = (
    _REPO_ROOT / "docs" / "project" / "architecture" / "CAPABILITY_CATALOG_AND_DISCOVERY.md"
)
_PLAN_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "plans"
    / "CAPABILITY_CATALOG_AND_DISCOVERY.md"
)

_CATALOG_NON_ENFORCEMENT_PHRASES = (
    "must not enforce isolation",
    "catalog discovery ≠ sandbox decision authority",
)

_STAGE12_MARKERS = (
    "trusted in-process",
    "must not enforce isolation",
    "no implicit fallback",
    "package loading",
    "execution isolation",
    "go/no-go",
    "marketplace source ≠ trust authority",
    "skill is not directly executable",
    "universalsandboxengine",
)

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _normalize_doc_text(text: str) -> str:
    return text.replace("*", "").lower()


def test_stage12_adr_exists_and_contains_normative_markers() -> None:
    assert _ADR_PATH.is_file(), f"missing Stage 12 ADR: {_ADR_PATH}"
    content = _normalize_doc_text(_read(_ADR_PATH))
    for marker in _STAGE12_MARKERS:
        assert marker.lower() in content, f"ADR missing marker: {marker}"


def test_stage12_docs_link_adr_and_freeze_catalog_non_enforcement() -> None:
    platform_plugins = _read(_PLATFORM_PLUGINS_DOC)
    catalog = _read(_CATALOG_DOC)
    plan = _read(_PLAN_DOC)

    assert "ADR-SEC-002" in platform_plugins
    assert "ADR-SEC-002" in catalog
    assert "Implemented assessment" in plan
    catalog_lower = _normalize_doc_text(catalog)
    assert any(phrase in catalog_lower for phrase in _CATALOG_NON_ENFORCEMENT_PHRASES)


def test_platform_plugin_trust_model_remains_trusted_in_process_only() -> None:
    assert list(PlatformPluginTrustModel) == [PlatformPluginTrustModel.TRUSTED_IN_PROCESS]


def test_wire_application_environment_signature_unchanged() -> None:
    signature = inspect.signature(wire_application_environment)
    assert "manifest" in signature.parameters
    assert "env" in signature.parameters
    assert "conformance_check" in signature.parameters


def test_capability_catalog_contracts_have_no_execution_posture_field() -> None:
    """Stage 12: no premature execution_posture schema until reuse threshold met."""
    contracts_root = _REPO_ROOT / "intergrax" / "contracts" / "capability_catalog"
    for path in contracts_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        if item.target.id == "execution_posture":
                            raise AssertionError(f"{path} defines execution_posture field")


def test_no_universal_isolation_engine_in_platform_layer() -> None:
    platform_root = _REPO_ROOT / "intergrax"
    forbidden_names = {
        "UniversalIsolationEngine",
        "UniversalSandboxEngine",
        "CapabilityExecutionEngine",
        "IsolationProviderRegistry",
    }
    for path in platform_root.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in forbidden_names:
                raise AssertionError(f"{path} defines forbidden class {node.name}")
