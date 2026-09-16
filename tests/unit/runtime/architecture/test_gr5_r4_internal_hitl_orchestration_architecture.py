# © Artur Czarnecki. All rights reserved.

"""GR-5-R4 architecture gates for internal HITL orchestration."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[4]
_CATEGORY_C = (
    _REPO / "intergrax/runtime/nexus/orchestration/intake_runner.py",
    _REPO / "intergrax/runtime/nexus/orchestration/planning_runner.py",
    _REPO / "intergrax/runtime/nexus/orchestration/graph_runner.py",
)

_FORBIDDEN_AUTHORITY = (
    "HumanPauseCoordinator.is_resumed",
    "HumanPauseCoordinator.clear_pause",
    "HumanPauseCoordinator.resolve_human_response(",
    "HumanPauseCoordinator.apply_pause",
)


def test_category_c_runners_do_not_use_task_projection_lifecycle_authority() -> None:
    for path in _CATEGORY_C:
        source = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_AUTHORITY:
            assert pattern not in source, f"{path.name} must not use {pattern} as authority"


def test_no_public_nexus_continuation_port_in_contracts() -> None:
    contracts = _REPO / "intergrax/contracts"
    for path in contracts.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "NexusContinuationPort" not in text
        assert "NexusHitlPort" not in text
