# © Artur Czarnecki. All rights reserved.

"""BND-03B — Tier-0 product manifest registry must not reach into Tier-3 dynamically."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_REGISTRY = _REPO / "intergrax" / "applications" / "_shared" / "product_manifest_registry.py"
_TIER3_SUFFIX = "_application"


def _tier3_application_roots(repo: Path) -> set[str]:
    apps = repo / "applications"
    if not apps.is_dir():
        return set()
    return {
        child.name
        for child in apps.iterdir()
        if child.is_dir() and child.name.endswith("_application") and (child / "__init__.py").is_file()
    }


def _registry_violations(path: Path, *, repo: Path) -> list[str]:
    rel = path.relative_to(repo).as_posix()
    text = path.read_text(encoding="utf-8")
    violations: list[str] = []
    if "importlib" in text:
        violations.append(f"{rel}: importlib reference")
    tier_roots = _tier3_application_roots(repo)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in {"import_module", "__import__"}:
                violations.append(f"{rel}: dynamic import call")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            value = node.value
            if "." in value:
                head = value.split(".", 1)[0]
                if head in tier_roots or head.endswith(_TIER3_SUFFIX):
                    violations.append(f"{rel}: Tier-3 module literal {value!r}")
            elif value in tier_roots or value.endswith(_TIER3_SUFFIX):
                violations.append(f"{rel}: Tier-3 name literal {value!r}")
    return violations


def test_product_manifest_registry_has_no_tier3_dynamic_edges() -> None:
    assert _registry_violations(_REGISTRY, repo=_REPO) == []


def test_negative_control_tier3_literal_detected(tmp_path: Path) -> None:
    fake_apps = tmp_path / "applications" / "legal_application"
    fake_apps.mkdir(parents=True)
    (fake_apps / "__init__.py").write_text("", encoding="utf-8")
    registry = tmp_path / "intergrax" / "applications" / "_shared" / "product_manifest_registry.py"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        'MODULE = "legal_application.manifest"\nimport importlib\nimportlib.import_module(MODULE)\n',
        encoding="utf-8",
    )
    violations = _registry_violations(registry, repo=tmp_path)
    assert any("legal_application" in item for item in violations)
    assert any("importlib" in item for item in violations)
