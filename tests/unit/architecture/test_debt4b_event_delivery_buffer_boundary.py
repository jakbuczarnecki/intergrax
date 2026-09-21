# © Artur Czarnecki. All rights reserved.

"""Architecture gate: no private BoundedEventSink enqueue reach-through in QoS tests."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_ATTRS = frozenset(
    {
        "_queue",
        "_quota_condition",
        "_pending_physical_enqueue",
    }
)
_SCAN_PATHS = (
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "observability"
    / "test_obs_delivery_qos_scale.py",
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "observability"
    / "test_bounded_delivery_p1b_r3_r2.py",
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "observability"
    / "test_enterprise_scale_resilience_w5_h_final_qualification.py",
    _REPO_ROOT
    / "tests"
    / "unit"
    / "runtime"
    / "observability"
    / "exporters"
    / "test_distributed_observability_transport.py",
)


def _private_attr_hits(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            hits.append(f"{path.name}:{node.lineno}:{node.attr}")
    return hits


def test_qos_tests_do_not_reach_private_enqueue_fields() -> None:
    all_hits: list[str] = []
    for path in _SCAN_PATHS:
        assert path.is_file(), f"missing scan target: {path}"
        all_hits.extend(_private_attr_hits(path))
    assert all_hits == []
