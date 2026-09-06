# © Artur Czarnecki. All rights reserved.

"""Architecture gates for causal evidence paging contract (DG-002 R1)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_CAUSAL_PERSISTENCE = (
    _REPO_ROOT / "intergrax" / "runtime" / "observability" / "causal_evidence_persistence.py"
)
_MEMORY_PERSISTENCE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "memory_causal_evidence_persistence.py"
)
_DOCUMENT_PERSISTENCE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "document_store_causal_evidence_persistence.py"
)


def _import_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0]
            )
    return names


def test_causal_evidence_persistence_contract_has_no_document_store_imports() -> None:
    imports = _import_names(_CAUSAL_PERSISTENCE)
    assert "intergrax" in imports
    forbidden = {
        "intergrax.integrations",
    }
    module_text = _CAUSAL_PERSISTENCE.read_text(encoding="utf-8")
    for prefix in forbidden:
        assert prefix not in module_text


def test_causal_evidence_backends_do_not_import_diagnostics() -> None:
    for path in (_MEMORY_PERSISTENCE, _DOCUMENT_PERSISTENCE):
        text = path.read_text(encoding="utf-8")
        assert "intergrax.runtime.diagnostics" not in text
