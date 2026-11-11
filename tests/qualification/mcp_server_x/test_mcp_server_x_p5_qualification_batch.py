# © Artur Czarnecki. All rights reserved.

"""MCP-SERVER-X P5 consolidated security and execution qualification batch."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.mcp_server_x.selection import MCP_SERVER_X_P5_SELECTION_BATCH

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONFCUTDIR = _REPO_ROOT / "tests" / "qualification" / "mcp_server_x"


def test_mcp_server_x_p5_consolidated_batch_passes() -> None:
    for target in MCP_SERVER_X_P5_SELECTION_BATCH:
        exit_code = pytest.main(
            [
                f"--confcutdir={_CONFCUTDIR}",
                str(_REPO_ROOT / target),
                "-q",
                "--tb=short",
            ],
        )
        assert exit_code == 0, f"P5 target failed: {target} (exit={exit_code})"
