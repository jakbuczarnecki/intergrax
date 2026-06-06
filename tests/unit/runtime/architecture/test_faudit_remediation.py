# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.applications.reference.harness_manifest_catalog import build_harness_reference_manifests
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState, audit_map_lifecycle_label
from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.decision_record import DecisionRecord
from intergrax.contracts.subtask_contract import SubtaskContract
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.llm_adapters.registry.model_router import ModelRouter
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.runtime.architecture.capability_graph_applications import catalog_application_manifests
from intergrax.runtime.interactions.envelope_intake import intake_envelope_to_task, intake_payload_to_envelope
from intergrax.runtime.nexus.errors.classifier import ErrorClassifier
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.task.task import Task
from intergrax.runtime.task_memory.retention_enforcement import should_forget_stm_record

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_tier0_manifest_catalog_has_no_applications_imports() -> None:
    manifests = catalog_application_manifests()
    assert {m.app_id for m in manifests} == {"lab", "legal", "research", "poc_template"}
    assert build_harness_reference_manifests() == manifests


def test_intergrax_no_applications_import_gate() -> None:
    script = REPO_ROOT / "scripts" / "check_intergrax_no_applications_imports.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_task_envelope_round_trip() -> None:
    envelope = TaskEnvelope(
        tenant_id="tenant-a",
        user_id="user-1",
        message="hello",
        agent_id="echo",
    )
    task = Task.from_envelope(envelope)
    assert task.to_envelope() == envelope


def test_intake_payload_envelope_parity() -> None:
    envelope = TaskEnvelope(
        tenant_id="tenant-b",
        user_id="worker-user",
        message="ping",
        agent_id="echo",
    )
    task_from_envelope = intake_envelope_to_task(envelope)
    task_roundtrip = Task.from_envelope(task_from_envelope.to_envelope())
    assert task_roundtrip.tenant_id == "tenant-b"
    assert task_roundtrip.user_id == "worker-user"
    assert task_roundtrip.message == "ping"
    assert task_roundtrip.agent_id == "echo"


def test_actor_identity_scope() -> None:
    actor = ActorIdentity(
        kind=ActorKind.SERVICE,
        actor_id="svc-1",
        tenant_id="tenant-a",
        permission_scopes=("tool:read",),
    )
    assert actor.allows_scope("tool:read") is True
    assert actor.allows_scope("tool:write") is False


def test_subtask_contract_safer_defaults() -> None:
    contract = SubtaskContract(child_agent_id="research")
    spec = contract.to_delegation_spec()
    assert spec.inherit_tool_policy is False
    assert spec.child_agent_id == "research"


def test_policy_pre_llm_and_pre_output_hooks() -> None:
    engine = PolicyEngine()
    pre_llm = engine.evaluate_pre_llm(tenant_id="t1", agent_id="echo", message_count=2)
    pre_out = engine.evaluate_pre_output(tenant_id="t1", agent_id="echo", output_chars=12)
    assert pre_llm.action.value == "allow"
    assert pre_out.action.value == "allow"


def test_model_router_policy_hint() -> None:
    from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

    primary = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o-mini")
    fallback = LLMProfile(provider=LLMProvider.OPENAI, model="gpt-4o")
    router = ModelRouter.from_profiles(primary, fallback=fallback, policy_route_hint="balanced")
    decision = router.resolve()
    assert decision.model == "gpt-4o"
    assert decision.routing_reason == "policy_hint_balanced"


def test_error_classifier_taxonomy() -> None:
    assert ErrorClassifier.classify(PermissionError("denied")) == RuntimeErrorCode.PERMISSION_ERROR
    assert ErrorClassifier.classify(ConnectionError("down")) == RuntimeErrorCode.DEPENDENCY_ERROR


def test_lifecycle_audit_map_labels() -> None:
    assert audit_map_lifecycle_label(AgentLifecycleState.PRODUCTION) == "certified"
    assert audit_map_lifecycle_label(AgentLifecycleState.EXPERIMENTAL) == "draft"


def test_data_classification_rules() -> None:
    assert DataClassification.RESTRICTED.requires_encryption() is True
    assert DataClassification.PUBLIC.allows_export() is True


def test_decision_record_shape() -> None:
    record = DecisionRecord(
        tenant_id="t1",
        task_id="task_1",
        run_id="run_1",
        agent_id="echo",
        step_id="step_1",
        decision_type="continue",
    )
    assert record.version == "decision_record.v1"


def test_stm_retention_enforcement() -> None:
    assert should_forget_stm_record(
        updated_at_utc="2000-01-01T00:00:00+00:00",
        retention_days=30,
        namespace="stm:session",
    ) is True
    assert should_forget_stm_record(
        updated_at_utc="2000-01-01T00:00:00+00:00",
        retention_days=30,
        namespace="ltm:user",
    ) is False
