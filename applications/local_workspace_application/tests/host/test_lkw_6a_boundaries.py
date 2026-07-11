# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_LKW_ROOT = Path(__file__).resolve().parents[2]


def test_no_lkw_specific_interaction_adapter_hierarchy() -> None:
    violations: list[str] = []
    for path in _LKW_ROOT.rglob("*.py"):
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "class Lkw" in text and "Interaction" in text:
            violations.append(str(path))
        if "class LocalWorkspace" in text and "InteractionAdapter" in text:
            violations.append(str(path))
    assert violations == []


def test_no_os_service_or_slack_socket_or_file_watcher_in_lkw_implementation() -> None:
    forbidden = (
        "Windows Service",
        "systemd",
        "launchd",
        "Socket Mode",
        "file watcher",
    )
    matches: list[str] = []
    for path in _LKW_ROOT.rglob("*.py"):
        if "tests" in path.parts or "docs" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                matches.append(f"{path}: {token}")
    assert matches == []
