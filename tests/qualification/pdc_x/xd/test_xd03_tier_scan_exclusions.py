# © Artur Czarnecki. All rights reserved.

"""XD-03 tier scanner: vendored docker runtime-context mirrors must not pollute the census."""

from __future__ import annotations

from pathlib import Path

import pytest
from tests.qualification.pdc_x.xd.test_xd_03_agent_governance import (
    REPO_ROOT,
    _collect_tool_executor_offender_paths,
    _is_vendored_application_docker_runtime_context,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("relative", "excluded"),
    [
        ("applications/example/docker/runtime-context/module.py", True),
        ("applications/example/host/module.py", False),
        ("agents/example/module.py", False),
        ("applications/example/docker/other/module.py", False),
    ],
)
def test_xd03_vendored_mirror_path_exclusion(relative: str, excluded: bool) -> None:
    path = REPO_ROOT / Path(*relative.split("/"))
    assert _is_vendored_application_docker_runtime_context(path) is excluded


def test_xd03_illegal_tool_executor_outside_mirror_is_detected() -> None:
    probe_root = REPO_ROOT / ".tmp" / "session" / "xd03-tier-scan" / "applications" / "probe_app" / "host"
    probe_root.mkdir(parents=True, exist_ok=True)
    illegal = probe_root / "bad_tool.py"
    illegal.write_text(
        "from intergrax.tools.tool_executor import ToolExecutor\n",
        encoding="utf-8",
    )
    try:
        offenders = _collect_tool_executor_offender_paths([illegal])
        assert offenders == [illegal.relative_to(REPO_ROOT).as_posix()]
    finally:
        illegal.unlink(missing_ok=True)


def test_xd03_legal_module_without_tool_executor_is_not_reported() -> None:
    probe_root = REPO_ROOT / ".tmp" / "session" / "xd03-tier-scan" / "applications" / "probe_app" / "host"
    probe_root.mkdir(parents=True, exist_ok=True)
    legal = probe_root / "clean.py"
    legal.write_text("x = 1\n", encoding="utf-8")
    try:
        assert _collect_tool_executor_offender_paths([legal]) == []
    finally:
        legal.unlink(missing_ok=True)
