# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
BRIDGE = (
    REPO
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "agent_governance_approval_pause_bridge.py"
)
HOST = (
    REPO
    / "intergrax"
    / "runtime"
    / "nexus"
    / "tools"
    / "continuation_aware_catalog_tool_host.py"
)
INVOKER = REPO / "intergrax" / "runtime" / "nexus" / "tools" / "invoker.py"
SIGNAL = REPO / "intergrax" / "contracts" / "agent_governance_approval_pause_signal.py"


def test_agent_bridge_lives_in_ee_l3_host_graph() -> None:
    host_source = HOST.read_text(encoding="utf-8")
    assert "agent_governance_approval_pause_bridge" in host_source
    assert "_materialize_agent_governance_pause" in host_source


def test_runtime_tool_invoker_does_not_import_agent_pause_bridge() -> None:
    source = INVOKER.read_text(encoding="utf-8")
    assert "agent_governance_approval_pause_bridge" not in source
    assert "HumanRequest" not in source


def test_no_second_agent_pause_store_types() -> None:
    for path in (BRIDGE, HOST):
        text = path.read_text(encoding="utf-8")
        assert "AgentGovernancePauseStore" not in text
        assert "AgentGovernanceContinuationStore" not in text


def test_bridge_does_not_create_grant() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    assert "AgentGovernanceHumanApprovalGrant(" not in source


def test_signal_contract_is_nexus_free() -> None:
    text = SIGNAL.read_text(encoding="utf-8")
    assert "runtime.nexus" not in text


def test_bridge_does_not_source_idempotency_from_approval_evidence_ref() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    assert "idempotency_key=authorization_request.approval_evidence_ref" not in source
    assert "idempotency_key=idempotency_key" in source or "idempotency_key=request.idempotency_key" in source


def test_host_uses_task_checkpoint_pause_projection() -> None:
    host_source = HOST.read_text(encoding="utf-8")
    assert "TaskAgentGovernancePauseProjectionAdapter" in host_source
    assert "task_checkpoint_store" in host_source
    assert "load_active_for_logical_invocation" in host_source


def test_bridge_bans_reflection_and_tigae() -> None:
    source = BRIDGE.read_text(encoding="utf-8")
    assert "ToolInvocationGovernanceApprovalEvidence" not in source
    assert "governance_approval_evidence" not in source
    assert "uca6c-scope" not in source
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in {"getattr", "setattr", "eval", "exec"}
