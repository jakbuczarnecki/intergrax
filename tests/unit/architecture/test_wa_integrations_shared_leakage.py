# © Artur Czarnecki. All rights reserved.

"""T-IGX-043-G (PT-WA, Part E): production Tier-2/3 modules do not import integrations._shared.

NT-RW-WA-01 (EQOA-A2-009 / A5-001). Production code reaches the in-process reference store and
the conformance helpers through the registry facades; ``integrations._shared`` stays internal.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[3]
_ROOTS = ("agents", "applications")
_FORBIDDEN = "intergrax.integrations._shared"

# Demo and proof modules that still import ``integrations._shared`` on purpose. The list is an
# explicit named tuple until ``PT-G01`` owns the exception list and replaces it (Q-WA-10).
_DEMO_PROOF_EXCEPTIONS = (
    "applications/attestation_demo/host/factory.py",
    "applications/attestation_demo/host/integration_wiring.py",
    "applications/attestation_demo/host/tool_wiring.py",
    "applications/local_workspace_application/model_runtime_proof/runner.py",
    "applications/local_workspace_application/model_runtime_proof/runtime.py",
)


def _is_production(path: Path) -> bool:
    parts = path.relative_to(_REPO).parts
    return "tests" not in parts and not path.name.startswith("test_") and path.name != "conftest.py"


def _imports_integrations_shared(tree: ast.AST) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == _FORBIDDEN or alias.name.startswith(_FORBIDDEN + ".") for alias in node.names):
                lines.append(node.lineno)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == _FORBIDDEN or node.module.startswith(_FORBIDDEN + "."):
                lines.append(node.lineno)
            elif node.module == "intergrax.integrations" and any(alias.name == "_shared" for alias in node.names):
                lines.append(node.lineno)
    return lines


def test_nt_rw_wa_01_production_modules_do_not_import_integrations_shared() -> None:
    violations: list[str] = []
    scanned = 0
    for root in _ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            if not _is_production(path):
                continue
            rel = path.relative_to(_REPO).as_posix()
            if rel in _DEMO_PROOF_EXCEPTIONS:
                continue
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=rel)
            violations.extend(f"{rel}:{line}" for line in _imports_integrations_shared(tree))
    assert scanned > 100
    assert violations == []
    for rel in _DEMO_PROOF_EXCEPTIONS:
        assert (_REPO / rel).is_file(), f"stale exception entry: {rel}"
