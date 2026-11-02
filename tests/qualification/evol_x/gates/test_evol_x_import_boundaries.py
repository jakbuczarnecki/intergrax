# © Artur Czarnecki. All rights reserved.

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_REPO = Path(__file__).resolve().parents[4]
_CHANGE_SAFETY = _REPO / "testing_support" / "change_safety"
_FORBIDDEN = ("tests", "tests.qualification")


def _violations_in_source(source: str, rel: str, forbidden_roots: tuple[str, ...] = _FORBIDDEN) -> list[str]:
    tree = ast.parse(source, filename=rel)
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in forbidden_roots or alias.name.startswith("tests."):
                    out.append(f"{rel}:{node.lineno}:{alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".", 1)[0]
            if root in forbidden_roots or node.module.startswith("tests."):
                out.append(f"{rel}:{node.lineno}:{node.module}")
    return out


def _violations_in_tree(path: Path) -> list[str]:
    return _violations_in_source(path.read_text(encoding="utf-8-sig"), path.relative_to(_REPO).as_posix())


def test_change_safety_does_not_import_tests_package() -> None:
    violations: list[str] = []
    for path in sorted(_CHANGE_SAFETY.rglob("*.py")):
        violations.extend(_violations_in_tree(path))
    assert violations == [], "; ".join(violations)


def test_import_boundary_detector_flags_synthetic_bad_import() -> None:
    assert _violations_in_source("from tests.qualification import x\n", "synthetic.py")
    assert _violations_in_source("import tests\n", "synthetic.py")
    assert _violations_in_source("import json\nfrom intergrax import x\n", "synthetic.py") == []


def test_tier_boundaries_agents_and_applications_do_not_import_change_safety() -> None:
    """IB: shipped tiers must not depend on qualification tooling (agents/ and applications/ source trees)."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "agents", "applications"], cwd=_REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    violations: list[str] = []
    for rel in tracked:
        if not rel.endswith(".py") or "/tests/" in rel or "/test_" in rel:
            continue
        try:
            source = (_REPO / rel).read_text(encoding="utf-8-sig")
        except OSError:
            continue
        if "change_safety" not in source:
            continue
        violations.extend(_violations_in_source(source, rel, forbidden_roots=("tests",)))
        if "testing_support.change_safety" in source and "import" in source:
            violations.append(rel)
    assert violations == [], "; ".join(violations[:10])


def test_contracts_roundtrip_enums() -> None:
    from testing_support.change_safety.contracts.enums import ImpactStatus

    assert ImpactStatus.COMPLETE.value == "COMPLETE"
