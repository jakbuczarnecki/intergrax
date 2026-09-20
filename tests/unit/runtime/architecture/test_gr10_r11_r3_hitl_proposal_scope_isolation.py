# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R3 — legacy human continuation proposal-scope isolation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import ExecutionContinuationLifecycleState
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotExecutor,
)
from intergrax.runtime.policy.mse_hitl_effect_gate import (
    EffectContinuationClassification,
    HumanGovernedProposalRelation,
    MseHitlEffectGateDisposition,
    classify_human_governed_proposal_relation,
    resolve_effect_continuation_context,
)
from intergrax.runtime.task.task import Task
from tests.unit.runtime.architecture.test_gr10_r11_r2_post_hitl_approval_evidence import (
    _ATTEMPT_ID,
    _EXECUTION_ID,
    _FakeContinuationPort,
    _OPERATION,
    _OPERATION_B,
    _RESOURCE,
    _RUN_ID,
    _SCOPE_1,
    _SCOPE_2,
    _SCOPE_DIGEST_1,
    _SCOPE_DIGEST_2,
    _TASK_ID,
    _TENANT,
    _ACTING,
    _correlation,
    _decision,
    _enforcement_request,
    _eval_gate,
    _grant,
    _seed_boundary,
)
from tests.unit.runtime.governance.gr3_test_support import bound_gr3_active_execution
from tests.unit.runtime.policy.test_g5c2b2b_governed_side_effect_reauthorization import (
    MutableRuntimePolicyEvaluator,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_GATE = _REPO / "intergrax/runtime/policy/mse_hitl_effect_gate.py"


def _side_effect(
    *,
    operation_id: str = _OPERATION,
    scope_id: str = _SCOPE_1,
    scope_digest: str = _SCOPE_DIGEST_1,
    resource: str | None = _RESOURCE,
) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action=operation_id,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=scope_id,
        side_effect_scope_digest=scope_digest,
        task_id=_TASK_ID,
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
        principal_id=_ACTING,
        tenant_id=_TENANT,
        resource=resource,
    )


def test_legacy_human_request_id_alone_is_correlation_insufficient() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=None,
        human_request_id="hr-legacy-r11r3",
    )
    side_effect = _side_effect()
    context = resolve_effect_continuation_context(
        port,
        side_effect=side_effect,
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.CORRELATION_INSUFFICIENT
    assert context.pending is not None
    relation = classify_human_governed_proposal_relation(
        context.pending,
        side_effect=side_effect,
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert relation is HumanGovernedProposalRelation.CORRELATION_INSUFFICIENT
    disposition, _ = _eval_gate(grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_same_execution_different_scope_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(
            side_effect_scope_id=_SCOPE_1,
            side_effect_scope_digest=_SCOPE_DIGEST_1,
        ),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_id=_SCOPE_2, scope_digest=_SCOPE_DIGEST_2),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION
    disposition, _ = _eval_gate(
        grant=None,
        port=port,
        scope_id=_SCOPE_2,
        scope_digest=_SCOPE_DIGEST_2,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_same_execution_different_operation_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(operation_id=_OPERATION),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(operation_id=_OPERATION_B),
        operation_id=_OPERATION_B,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION
    disposition, _ = _eval_gate(operation_id=_OPERATION_B, grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_same_execution_different_resource_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(),
    )
    other_resource = "resource-r11r3-other"
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(resource=other_resource),
        operation_id=_OPERATION,
        resource_scope=other_resource,
    )
    assert context.classification is EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION
    relation = classify_human_governed_proposal_relation(
        context.pending,  # type: ignore[arg-type]
        side_effect=_side_effect(resource=other_resource),
        operation_id=_OPERATION,
        resource_scope=other_resource,
    )
    assert relation is HumanGovernedProposalRelation.UNRELATED_HUMAN_CONTINUATION


def test_same_execution_different_digest_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(
            side_effect_scope_id=_SCOPE_1,
            side_effect_scope_digest=_SCOPE_DIGEST_1,
        ),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_id=_SCOPE_1, scope_digest=_SCOPE_DIGEST_2),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION
    disposition, _ = _eval_gate(
        grant=None,
        port=port,
        scope_id=_SCOPE_1,
        scope_digest=_SCOPE_DIGEST_2,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_exact_proposal_match_post_hitl_missing_grant_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.POST_HITL_RESUMED
    disposition, _ = _eval_gate(grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_exact_proposal_match_post_hitl_matching_grant_proceeds() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, task = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    assert task.runtime.governance.governed_continuation_grant is None


def test_legacy_correlation_less_does_not_contaminate_new_effect() -> None:
    """Same execution + human_request_id + no correlation → effect B ordinary."""
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=None,
        human_request_id="hr-old-effect-a",
        reason=ContinuationReason.COMPLIANCE,
    )
    disposition, _ = _eval_gate(
        grant=None,
        port=port,
        scope_id=_SCOPE_2,
        scope_digest=_SCOPE_DIGEST_2,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED


async def test_topology_cross_slot_isolation_legacy_and_correlated() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator, operation_id=_OPERATION_B)
    effects = [0]
    port = _FakeContinuationPort()
    # Slot A HITL history (different operation correlation) on same execution.
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(operation_id=_OPERATION),
    )

    class _Inner:
        async def execute_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "slot-b-done"

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    executor = GovernedOrchestrationSlotExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _s, _p: _enforcement_request(
            membership,
            operation_id=_OPERATION_B,
            scope_id=_SCOPE_2,
            scope_digest=_SCOPE_DIGEST_2,
        ),
        continuation_port=port,
    )
    governed = ActiveGovernedExecutionTask()
    token = governed.bind(task)
    try:
        with bound_gr3_active_execution(
            run_id=_RUN_ID,
            attempt_id=_ATTEMPT_ID,
            execution_id=_EXECUTION_ID,
        ):
            result = await executor.execute_slot(
                slot_id=OrchestrationSlotId("slot-b-independent"),
                payload=object(),
            )
    finally:
        governed.reset(token)
    assert result == "slot-b-done"
    assert effects[0] == 1


def test_static_gate_no_human_request_id_fallback_to_hitl() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "classify_human_governed_proposal_relation" in source
    assert "CORRELATION_INSUFFICIENT" in source
    assert "UNRELATED_HUMAN_CONTINUATION" in source
    assert "human_request_id alone" in source.lower() or (
        "human_request_id`` alone" in source
    )
    # Forbidden logical equivalent of R11-R2 legacy fallback.
    tree = ast.parse(source, filename=str(_GATE))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Return):
            continue
        # Detect: return pending.human_request_id is not None
        value = node.value
        if not isinstance(value, ast.Compare):
            continue
        left = value.left
        if isinstance(left, ast.Attribute) and left.attr == "human_request_id":
            for comparator in value.comparators:
                if isinstance(comparator, ast.Constant) and comparator.value is None:
                    # Allow `is None` checks; forbid `is not None` as sole return proof.
                    for op in value.ops:
                        assert not isinstance(op, ast.IsNot), (
                            "human_request_id is not None must not be positive HITL proof"
                        )


def test_static_gate_no_post_hitl_from_human_request_id_without_correlation() -> None:
    source = _GATE.read_text(encoding="utf-8")
    # POST_HITL_* only after MATCHED_HITL_PROPOSAL path (lifecycle check after relation).
    assert "MATCHED_HITL_PROPOSAL" in source
    assert "return pending.human_request_id is not None" not in source
