# © Artur Czarnecki. All rights reserved.

"""IDEAL-L3 depth gate — critical L2→L3 uplift evidence (Phase IDEAL-L3 W1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState
from intergrax.contracts.data_classification import DataClassification
from intergrax.contracts.delegation import DelegationSpec
from intergrax.contracts.subtask_contract import SubtaskContract
from intergrax.contracts.task_envelope import TaskEnvelope, TaskRiskTier, TaskSlaClass
from intergrax.runtime.architecture.data_classification_enforcement import (
    DataClassificationPolicyError,
    assert_data_export_allowed,
)
from intergrax.runtime.interactions.actor_resolution import (
    narrow_delegation_scopes,
    resolve_actor_from_envelope,
)
from intergrax.runtime.nexus.budget.production_budget_policy import (
    ensure_production_run_budget,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.errors.classifier import ErrorClassifier
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.errors.harness_error_taxonomy import (
    HarnessErrorFamily,
    family_for_code,
    is_dependency_failure,
    is_quality_failure,
    recovery_for_code,
)
from intergrax.runtime.observability.harness_slos import list_harness_slos, slo_ids
from intergrax.runtime.registry.agent_routing_policy import evaluate_agent_routing
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_ideal_l3_error_taxonomy_covers_ideal_families() -> None:
    assert family_for_code(RuntimeErrorCode.QUALITY_ERROR) is HarnessErrorFamily.QUALITY
    assert family_for_code(RuntimeErrorCode.DEPENDENCY_ERROR) is HarnessErrorFamily.DEPENDENCY
    assert family_for_code(RuntimeErrorCode.RUNTIME_ERROR) is HarnessErrorFamily.RUNTIME
    assert recovery_for_code(RuntimeErrorCode.DEPENDENCY_ERROR).value == "retry_with_backoff"
    assert is_quality_failure(RuntimeErrorCode.QUALITY_ERROR)
    assert is_dependency_failure(RuntimeErrorCode.TIMEOUT)
    assert ErrorClassifier.classify(ConnectionError("down")) is RuntimeErrorCode.DEPENDENCY_ERROR
    assert ErrorClassifier.classify(RuntimeError("race")) is RuntimeErrorCode.RUNTIME_ERROR


def test_ideal_l3_data_classification_enforcement() -> None:
    assert_data_export_allowed(DataClassification.PUBLIC)
    with pytest.raises(DataClassificationPolicyError):
        assert_data_export_allowed(DataClassification.CONFIDENTIAL, external_llm=True)


def test_ideal_l3_production_run_budget_auto_fill() -> None:
    from testing_support.builder import FakeLLMAdapter

    config = RuntimeConfig(
        llm_adapter=FakeLLMAdapter(fixed_text="ok"),
        production_mode=True,
        run_budget=None,
    )
    ensure_production_run_budget(config)
    assert config.run_budget is not None
    assert config.run_budget.max_total_tokens is not None


def test_ideal_l3_task_envelope_sla_and_risk_fields() -> None:
    envelope = TaskEnvelope(
        tenant_id="t1",
        user_id="u1",
        message="hi",
        sla_class=TaskSlaClass.BATCH,
        risk_tier=TaskRiskTier.HIGH,
        constraints={"max_cost_usd": 1.0},
    )
    task = Task.from_envelope(envelope)
    roundtrip = task.to_envelope()
    assert roundtrip.sla_class is TaskSlaClass.BATCH
    assert roundtrip.risk_tier is TaskRiskTier.HIGH
    assert roundtrip.constraints["max_cost_usd"] == 1.0


def test_ideal_l3_actor_resolution_and_scope_narrowing() -> None:
    envelope = TaskEnvelope(
        tenant_id="t1",
        user_id="u1",
        message="go",
    ).with_actor(actor_kind=ActorKind.SERVICE.value, actor_id="svc-ops")
    actor = resolve_actor_from_envelope(envelope)
    assert actor.kind is ActorKind.SERVICE
    parent = ActorIdentity(
        kind=ActorKind.USER,
        actor_id="u1",
        tenant_id="t1",
        permission_scopes=("read", "write"),
    )
    delegation = DelegationSpec(
        child_agent_id="echo",
        permission_scopes=("read", "admin"),
    )
    assert narrow_delegation_scopes(parent, delegation) == ("read",)


def test_ideal_l3_subtask_contract_safe_defaults() -> None:
    contract = SubtaskContract(child_agent_id="echo")
    assert contract.inherit_tool_policy is False
    spec = contract.to_delegation_spec()
    assert spec.inherit_tool_policy is False


def test_ideal_l3_retired_agent_not_routable_in_production() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract

    contract = AgentContract(
        id="retired-agent",
        name="Retired",
        description="x",
        lifecycle_state=AgentLifecycleState.RETIRED,
    )
    decision = evaluate_agent_routing(contract, production_mode=True)
    assert decision.routable is False


def test_ideal_l3_harness_slo_catalog() -> None:
    slos = list_harness_slos()
    assert len(slos) >= 5
    assert "harness.run.availability" in slo_ids()


def test_ideal_l3_umbrella_gate_script() -> None:
    script = REPO_ROOT / "scripts" / "check_ideal_harness_l3_gates.py"
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
