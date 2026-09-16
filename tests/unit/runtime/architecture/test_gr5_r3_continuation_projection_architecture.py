# © Artur Czarnecki. All rights reserved.

"""GR-5-R3 architecture gates — projection contract and coordinator boundaries."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation_projection import (
    ExecutionContinuationProjectionSink,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PAUSE_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "human" / "pause.py"
_PROJECTION_CONTRACT = (
    _REPO_ROOT / "intergrax" / "contracts" / "execution_continuation_projection.py"
)


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_projection_contract_no_runtime_task_imports() -> None:
    modules = _collect_imports(_PROJECTION_CONTRACT)
    forbidden = [m for m in modules if m.startswith("intergrax.runtime")]
    assert forbidden == []


def test_human_pause_coordinator_no_concrete_continuation_service() -> None:
    source = _PAUSE_MODULE.read_text(encoding="utf-8")
    assert "ExecutionContinuationService" not in source
    assert "InMemoryExecutionContinuationStateStore" not in source


def test_no_public_nexus_projection_contract() -> None:
    source = _PROJECTION_CONTRACT.read_text(encoding="utf-8")
    for forbidden in (
        "NexusPort",
        "NexusContinuationPort",
        "PublicNexusFacade",
    ):
        assert forbidden not in source


def test_projection_sink_protocol_replaceable() -> None:
    class _Sink:
        def project(self, pending):  # noqa: ANN001
            return None

    assert isinstance(_Sink(), ExecutionContinuationProjectionSink)


def test_revision_domain_scoped_to_continuation_id() -> None:
    source = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "task"
        / "execution_continuation_projection.py"
    ).read_text(encoding="utf-8")
    assert "last_id == pending.continuation_id" in source
    assert "continuation_projection_allows_replacement" in source


def test_active_continuation_replacement_blocked_in_prepare() -> None:
    source = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "task"
        / "execution_continuation_projection.py"
    ).read_text(encoding="utf-8")
    assert "task already projects a different continuation_id" in source
    assert "continuation_projection_allows_replacement" in source


def test_completed_continuation_replacement_allowed_in_prepare() -> None:
    source = (
        _REPO_ROOT
        / "intergrax"
        / "runtime"
        / "task"
        / "execution_continuation_projection.py"
    ).read_text(encoding="utf-8")
    assert "replace_completed_continuation" in source
    assert "ExecutionContinuationLifecycleState.RESUMED" in source


def test_canonical_resolution_does_not_call_task_hitl_before_apply() -> None:
    source = _PAUSE_MODULE.read_text(encoding="utf-8")
    canonical_start = source.index("def resolve_human_response_and_apply_canonical")
    canonical_block = source[canonical_start : canonical_start + 4500]
    apply_idx = canonical_block.index("continuation.apply_resolution")
    hitl_assign = "gov.hitl_resolution ="
    assert hitl_assign not in canonical_block[:apply_idx]
    resolve_call = "resolve_human_response("
    assert resolve_call not in canonical_block
