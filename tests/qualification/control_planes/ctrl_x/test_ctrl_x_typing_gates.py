# © Artur Czarnecki. All rights reserved.

"""CTRL-X-R1 typing scope and wide-pyright provenance gates."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.qualification.control_planes.ctrl_x.typing_scope import (
    CTRL_X_SEMANTIC_BOUNDARY_MODULES,
    ctrl_x_semantic_boundary_module_paths,
)
from tests.qualification.control_planes.ctrl_x.wide_pyright_provenance import (
    CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_GROUPS,
    CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_TOTAL,
)
from tests.qualification.control_planes.ctrl_x.wide_pyright_scope import (
    CTRL_X_WIDE_PYRIGHT_FILE_PATHS,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ctrl_x_each_plane_has_typing_target() -> None:
    for plane in (f"CX-{i:02d}" for i in range(1, 13)):
        assert plane in CTRL_X_SEMANTIC_BOUNDARY_MODULES
        assert CTRL_X_SEMANTIC_BOUNDARY_MODULES[plane]


def test_ctrl_x_semantic_boundary_pyright_zero_errors() -> None:
    targets = ctrl_x_semantic_boundary_module_paths()
    proc = subprocess.run(
        ["uv", "run", "pyright", *targets],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_ctrl_x_wide_pyright_provenance_accounts_for_all_diagnostics() -> None:
    accounted = sum(group["diagnostic_count"] for group in CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_GROUPS)
    assert accounted == CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_TOTAL
    assert CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_TOTAL > 0
    assert len(CTRL_X_WIDE_PYRIGHT_FILE_PATHS) >= 50
    blockers = [
        group
        for group in CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_GROUPS
        if group.get("ctrl_x_impact") == "CTRL-X BLOCKER"
    ]
    assert blockers == []
    for group in CTRL_X_WIDE_PYRIGHT_DIAGNOSTIC_GROUPS:
        assert group.get("semantic_plane")
        assert group.get("reason")
        assert group.get("ctrl_x_impact")
