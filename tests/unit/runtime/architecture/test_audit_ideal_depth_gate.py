# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL depth gate evidence (Band 2az)."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot
from intergrax.applications._shared.registry_snapshot_store import persist_registry_snapshot
from intergrax.applications._shared.replay_routes import create_replay_router
from intergrax.contracts.reasoning_profile import ReasoningProfile
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.memory.org_memory_scope import OrgMemoryScope
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_output_policy_bridge import apply_pre_output_policy
from intergrax.runtime.task.task import Task

pytestmark = pytest.mark.gate

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_audit_ideal_3_1_envelope_runtime_roundtrip() -> None:
    envelope = TaskEnvelope(tenant_id="t1", user_id="u1", message="hi", agent_id="echo")
    request = RuntimeRequest.from_envelope(envelope)
    assert request.tenant_id == "t1"
    assert request.message == "hi"
    assert Task.from_envelope(envelope).to_envelope().message == "hi"


def test_audit_ideal_5_1_pre_output_policy() -> None:
    engine = PolicyEngine()
    task = Task(tenant_id="t1", user_id="u1", agent_id="echo", message="m")
    answer, decision = apply_pre_output_policy(engine, task, answer="valid answer")
    assert answer == "valid answer"
    assert decision.action.value == "allow"
    blocked, _ = apply_pre_output_policy(engine, task, answer="")
    assert blocked.startswith("[POLICY_BLOCKED]")


def test_audit_ideal_7_1_reasoning_profile_exists() -> None:
    profile = ReasoningProfile()
    assert profile.planner_prompt_id == "nexus_task_planner"


def test_audit_ideal_15_1_org_memory_scopes() -> None:
    assert OrgMemoryScope.ORG_PROFILE.value == "org_profile"
    assert len(list(OrgMemoryScope)) == 3


def test_audit_ideal_19_1_registry_snapshot_store(tmp_path: Path) -> None:
    snapshot = HarnessRegistrySnapshot(
        integration_profile=None,
        tool_registry=None,
        skill_registry=None,
        prompt_registry=None,
        policy_bundle=None,
    )
    sid = persist_registry_snapshot(
        snapshot, host_id="lab", db_path=tmp_path / "registry.db", snapshot_id="snap_test"
    )
    assert sid == "snap_test"


def test_audit_ideal_27_2_replay_router() -> None:
    router = create_replay_router(enabled=True)
    assert any(route.path.endswith("/replay") for route in router.routes)


def test_audit_ideal_30_1_ecp_architecture_synced() -> None:
    arch = REPO_ROOT / "docs" / "architecture" / "ELASTIC_CAPACITY_AND_SCALING.md"
    text = arch.read_text(encoding="utf-8")
    assert "Harness elastic control loop" in text
    assert "L3" in text


def test_audit_ideal_deferred_register() -> None:
    register = REPO_ROOT / "docs" / "plan" / "AUDIT_IDEAL_2026.md"
    text = register.read_text(encoding="utf-8")
    for task_id in ("28.3", "28.4", "5.3", "21.3"):
        assert f"AUDIT-IDEAL-{task_id}" in text
        assert "Deferred" in text
