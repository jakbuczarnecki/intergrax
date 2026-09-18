# © Artur Czarnecki. All rights reserved.

"""GR-10-R2-R1 — PRE_MODEL runtime identity conformance gates."""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.contracts.runtime_policy_context import PreModelPhase, PreModelPolicyContext
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.execution.inference import InferenceExecutor
from intergrax.runtime.execution.request import ExecutionRequest
from intergrax.runtime.governance.active_execution_governance_identity import (
    ActiveExecutionGovernanceIdentity,
    bind_active_execution_governance_identity,
    reset_active_execution_governance_identity,
)
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.runtime.policy.pre_model_policy_evaluation import (
    PreModelPolicyConfigurationError,
    enforce_pre_model_before_structured_inference,
)
from intergrax.runtime.policy.pre_model_principal import (
    principal_id_for_orchestration_task,
    require_orchestration_pre_model_governance_scope,
    resolve_agentic_pre_model_scope,
)
from intergrax.runtime.task.task import Task
from testing_support.inference_governance_wiring import (
    TEST_INFERENCE_PRINCIPAL_ID,
    TEST_INFERENCE_TENANT_ID,
    TEST_INFERENCE_WORKSPACE_ID,
)
from tests.unit.runtime.execution.test_inference_executor import (
    RiskAssessment,
    StructuredTestAdapter,
    _risk_request,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
PRE_MODEL_EVAL_PATH = REPO_ROOT / "intergrax/runtime/policy/pre_model_policy_evaluation.py"
PRE_MODEL_PRINCIPAL_PATH = REPO_ROOT / "intergrax/runtime/policy/pre_model_principal.py"

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _bind_governance(
    *,
    tenant_id: str = TEST_INFERENCE_TENANT_ID,
    workspace_id: str = TEST_INFERENCE_WORKSPACE_ID,
    principal_id: str = TEST_INFERENCE_PRINCIPAL_ID,
):
    return bind_active_execution_governance_identity(
        ActiveExecutionGovernanceIdentity(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            principal_id=principal_id,
        ),
    )


@pytest.mark.asyncio
async def test_inference_pre_model_requires_active_governance_identity() -> None:
    adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
    identity_token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    try:
        executor = InferenceExecutor(adapter, policy_engine=PolicyEngine())
        with pytest.raises(PreModelPolicyConfigurationError, match="governance identity"):
            await executor.execute(_risk_request())
    finally:
        reset_active_execution_identity(identity_token)
    assert adapter.generate_structured_calls == 0


def test_orchestration_tenant_mismatch_fails_closed() -> None:
    token = _bind_governance()
    try:
        task = Task(
            tenant_id="tenant-other",
            user_id=TEST_INFERENCE_PRINCIPAL_ID,
            message="m",
        )
        task.metadata["workspace_id"] = TEST_INFERENCE_WORKSPACE_ID
        with pytest.raises(PreModelPolicyConfigurationError, match="mismatch"):
            principal_id_for_orchestration_task(task)
    finally:
        reset_active_execution_governance_identity(token)


def test_orchestration_requires_active_governance_identity() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="m")
    with pytest.raises(PreModelPolicyConfigurationError, match="governance identity"):
        principal_id_for_orchestration_task(task)


def test_orchestration_task_principal_mismatch_fails_closed() -> None:
    token = _bind_governance(principal_id="principal-a")
    try:
        task = Task(tenant_id=TEST_INFERENCE_TENANT_ID, user_id="principal-b", message="m")
        task.metadata["workspace_id"] = TEST_INFERENCE_WORKSPACE_ID
        with pytest.raises(PreModelPolicyConfigurationError, match="mismatch"):
            principal_id_for_orchestration_task(task)
    finally:
        reset_active_execution_governance_identity(token)


def test_orchestration_active_identity_matches_task_projection() -> None:
    token = _bind_governance()
    try:
        task = Task(
            tenant_id=TEST_INFERENCE_TENANT_ID,
            user_id=TEST_INFERENCE_PRINCIPAL_ID,
            message="m",
        )
        task.metadata["workspace_id"] = TEST_INFERENCE_WORKSPACE_ID
        assert principal_id_for_orchestration_task(task) == TEST_INFERENCE_PRINCIPAL_ID
    finally:
        reset_active_execution_governance_identity(token)


def test_orchestration_canonical_identity_mismatch_fails_closed() -> None:
    token = _bind_governance(principal_id="principal-a")
    try:
        task = Task(
            tenant_id=TEST_INFERENCE_TENANT_ID,
            user_id="principal-a",
            message="m",
            canonical_identity=RequestIdentity(
                tenant_id=TEST_INFERENCE_TENANT_ID,
                user_id="principal-b",
            ),
        )
        task.metadata["workspace_id"] = TEST_INFERENCE_WORKSPACE_ID
        with pytest.raises(PreModelPolicyConfigurationError, match="mismatch"):
            require_orchestration_pre_model_governance_scope(task)
    finally:
        reset_active_execution_governance_identity(token)


def test_agentic_production_requires_active_governance_identity() -> None:
    with pytest.raises(PreModelPolicyConfigurationError, match="governance identity"):
        resolve_agentic_pre_model_scope(
            tenant_id="t1",
            workspace_id="w1",
            request_principal_id="p1",
            production_mode=True,
        )


def test_agentic_active_identity_mismatch_fails_closed() -> None:
    token = _bind_governance(principal_id="principal-a")
    try:
        with pytest.raises(PreModelPolicyConfigurationError, match="mismatch"):
            resolve_agentic_pre_model_scope(
                tenant_id=TEST_INFERENCE_TENANT_ID,
                workspace_id=TEST_INFERENCE_WORKSPACE_ID,
                request_principal_id="principal-b",
                production_mode=False,
            )
    finally:
        reset_active_execution_governance_identity(token)


@pytest.mark.asyncio
async def test_step_kernel_production_empty_principal_fails_closed() -> None:
    from intergrax.agents.authoring.step_outcome import StepOutcome
    from intergrax.contracts.agent_step_context import AgentStepContext
    from testing_support.builder import kernel_step_test_scope

    with kernel_step_test_scope("gr10-r2-r1-missing-principal") as (task_id, run_id):
        kernel_ctx = StepKernelContext(
            agent_id="demo",
            task_id=task_id,
            run_id=run_id,
            principal_id="",
            production_mode=True,
            policy_engine=PolicyEngine(),
        )
        step_ctx = AgentStepContext(step_index=0)
        outcome = StepOutcome.continue_with({"phase": "plan"})
        record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
        assert record.policy_pre is not None
        assert record.policy_pre.action is PolicyAction.DENY
        assert record.policy_pre.policy_rule_id == "kernel.missing_principal_id"


def test_no_evidence_authority_in_pre_model_eval_module() -> None:
    text = PRE_MODEL_EVAL_PATH.read_text(encoding="utf-8")
    assert "peek_active_execution_evidence_context" not in text


def test_no_lineage_authority_in_pre_model_eval_module() -> None:
    text = PRE_MODEL_EVAL_PATH.read_text(encoding="utf-8")
    assert "peek_active_execution_lineage" not in text


def test_orchestration_principal_helper_no_task_user_id_fallback() -> None:
    text = PRE_MODEL_PRINCIPAL_PATH.read_text(encoding="utf-8")
    fn = text.split("def principal_id_for_orchestration_task", 1)[1].split("\ndef ", 1)[0]
    assert "task.user_id" not in fn or "validate_governance_identity_projection" in fn
    assert 'return ""' not in fn


def test_orchestration_principal_helper_no_canonical_identity_fallback() -> None:
    text = PRE_MODEL_PRINCIPAL_PATH.read_text(encoding="utf-8")
    fn = text.split("def principal_id_for_orchestration_task", 1)[1].split("\ndef ", 1)[0]
    assert "principal_id_from_request_identity(task.canonical_identity)" not in fn


class _RecordingPreModelRuntime:
    def evaluate_pre_llm(
        self, *, tenant_id, principal_id, agent_id=None, message_count, context=None
    ):
        self.last_principal = principal_id
        return PolicyDecision(
            action=PolicyAction.ALLOW,
            reason="allow",
            policy_rule_id="test.allow",
        )


def test_enforce_pre_model_evidence_uses_same_identity_as_policy() -> None:
    token = _bind_governance()
    adapter = StructuredTestAdapter(parsed_output=RiskAssessment(risk="low"))
    runtime = _RecordingPreModelRuntime()
    try:
        enforce_pre_model_before_structured_inference(
            PolicyEngine(runtime=runtime),
            None,
            adapter=adapter,
            messages=(ChatMessage(role="user", content="x"),),
            inference_profile_id=None,
        )
        assert runtime.last_principal == TEST_INFERENCE_PRINCIPAL_ID
    finally:
        reset_active_execution_governance_identity(token)
