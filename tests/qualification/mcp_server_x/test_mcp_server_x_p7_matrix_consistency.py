# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P7 matrix consistency (MCP-P7-09, AC-P7-01 and AC-P7-02).

Static checks, no server and no network: every MCP-ACC-01..18 has at least one
``test_acc_<NN>_*`` function in the P7 files, every P7 module states its level, L2 modules say
they are not a physical transport test, and the record-only rows (19, 20) have their evidence.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from tests.qualification.mcp_server_x.p7_selection import (
    MCP_SERVER_X_P7_NEW_QUALIFICATION,
    MCP_SERVER_X_P7_RECORD_ONLY_ACC,
    MCP_SERVER_X_P7_REGRESSION_SELECTION,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ACC = re.compile(r"^test_acc_(\d{2})_")
_PACKAGE = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "mcp_server_x"
    / "MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md"
)


def _parse(rel: str) -> ast.Module:
    path = _REPO_ROOT / rel
    assert path.is_file(), f"P7 file missing: {rel}"
    return ast.parse(path.read_text(encoding="utf-8"))


def _acc_ids() -> dict[int, list[str]]:
    found: dict[int, list[str]] = {}
    for rel in MCP_SERVER_X_P7_NEW_QUALIFICATION:
        for node in ast.walk(_parse(rel)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                match = _ACC.match(node.name)
                if match:
                    found.setdefault(int(match.group(1)), []).append(f"{rel}::{node.name}")
    return found


def test_acc_matrix_every_mcp_acc_01_to_18_has_a_p7_test() -> None:
    found = _acc_ids()
    missing = [n for n in range(1, 19) if not found.get(n)]
    assert not missing, f"MCP-ACC rows without a P7 test_acc_<NN>_ function: {missing}"


def test_acc_matrix_p7_test_ids_are_unique() -> None:
    names = [name for ids in _acc_ids().values() for name in ids]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("rel", MCP_SERVER_X_P7_NEW_QUALIFICATION)
def test_acc_matrix_every_p7_module_states_its_level(rel: str) -> None:
    doc = ast.get_docstring(_parse(rel)) or ""
    levels = [lvl for lvl in ("L1", "L2", "L3") if lvl in doc]
    assert levels, f"{rel}: module docstring must state L1, L2 or L3"
    if "L2" in doc and "L3" not in doc:
        assert "not a physical transport test" in doc.lower().replace("-", " ") or (
            "not a physical transport" in doc.lower()
        ), f"{rel}: an L2 module must say it is not a physical transport test"


def test_acc_matrix_regression_selection_files_exist() -> None:
    missing = [rel for rel in MCP_SERVER_X_P7_REGRESSION_SELECTION if not (_REPO_ROOT / rel).is_file()]
    assert not missing, f"regression selection lists missing files: {missing}"


def test_acc_matrix_record_only_rows_are_19_and_20() -> None:
    assert sorted(MCP_SERVER_X_P7_RECORD_ONLY_ACC) == [19, 20]


def test_acc_20_exact_sha_package_record_exists_and_pins_a_test_execution_sha() -> None:
    assert _PACKAGE.is_file(), "MCP_SERVER_X_P7_QUALIFICATION_PACKAGE.md must exist (MCP-ACC-20)"
    text = _PACKAGE.read_text(encoding="utf-8")
    assert re.search(r"TEST_EXECUTION_SHA[^`\n]*`[0-9a-f]{40}`", text), "package must pin a 40-hex TEST_EXECUTION_SHA"
    for n in range(1, 21):
        assert f"MCP-ACC-{n:02d}" in text, f"package record has no row for MCP-ACC-{n:02d}"


def test_acc_19_known_baseline_red_files_exist_and_are_excluded_from_the_regression_selection() -> None:
    from tests.qualification.mcp_server_x.p7_selection import MCP_SERVER_X_P7_KNOWN_BASELINE_RED

    assert len(MCP_SERVER_X_P7_KNOWN_BASELINE_RED) == 3
    for rel in MCP_SERVER_X_P7_KNOWN_BASELINE_RED:
        assert (_REPO_ROOT / rel).is_file(), rel
        assert rel not in MCP_SERVER_X_P7_REGRESSION_SELECTION, rel
