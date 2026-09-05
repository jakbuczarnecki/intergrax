# © Artur Czarnecki. All rights reserved.

"""NPSC-3B: LKW active public surface must not depend on NexusLoop."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_LKW_ROOT = _REPO_ROOT / "applications" / "local_workspace_application"

_FORBIDDEN_TOKENS: tuple[str, ...] = (
    "NexusLoop",
    "nexus_loop",
    "resolve_harness_host_nexus_loop_legacy",
)

_GATE_PATHS: tuple[Path, ...] = (
    _LKW_ROOT / "host" / "task_executor.py",
    *_LKW_ROOT.glob("serving/**/*.py"),
    *_LKW_ROOT.glob("mcp/**/*.py"),
)


def _violations_for_path(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    rel = path.relative_to(_REPO_ROOT).as_posix()
    return [
        f"{rel}: forbidden token {token!r}"
        for token in _FORBIDDEN_TOKENS
        if token in source
    ]


@pytest.mark.parametrize("path", _GATE_PATHS, ids=lambda path: path.relative_to(_LKW_ROOT).as_posix())
def test_npsc3b_lkw_public_surface_has_no_nexus_tokens(path: Path) -> None:
    assert _violations_for_path(path) == []
