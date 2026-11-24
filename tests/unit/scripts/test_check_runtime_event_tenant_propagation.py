# © Artur Czarnecki. All rights reserved.

"""Unit tests for the RuntimeEvent tenant propagation maintenance gate."""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.maintenance.check_runtime_event_tenant_propagation import (
    EMITTER_FILES,
    REPO_ROOT,
    _check_file,
    main,
)

pytestmark = [pytest.mark.unit]

_SCRATCH = REPO_ROOT / ".tmp" / "session" / "eqo-w1-a4006"


@pytest.fixture
def scratch_py() -> Path:
    _SCRATCH.mkdir(parents=True, exist_ok=True)
    return _SCRATCH


def test_runtime_event_without_tenant_id_is_violation(scratch_py: Path) -> None:
    target = scratch_py / "missing_tenant_kwarg.py"
    target.write_text(
        "def emit() -> None:\n    RuntimeEvent(event_type=1)\n",
        encoding="utf-8",
    )
    violations = _check_file(target)
    assert len(violations) == 1
    assert "RuntimeEvent missing tenant_id=" in violations[0]


def test_missing_emitter_path_is_violation() -> None:
    stale = REPO_ROOT / "intergrax/agents/uaep.py"
    assert not stale.is_file()
    violations = _check_file(stale)
    assert violations == ["missing emitter file: intergrax/agents/uaep.py"]


def test_emitter_files_includes_uaep_executor() -> None:
    uaep_path = REPO_ROOT / "intergrax/runtime/nexus/uaep/uaep_executor.py"
    assert uaep_path in EMITTER_FILES


def test_configured_emitter_paths_exist_and_main_exits_zero() -> None:
    for path in EMITTER_FILES:
        assert path.is_file(), f"configured emitter missing: {path}"
    assert main() == 0
