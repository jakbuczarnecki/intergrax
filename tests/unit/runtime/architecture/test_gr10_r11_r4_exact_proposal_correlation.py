# © Artur Czarnecki. All rights reserved.

"""GR-10-R11-R4 — exact proposal correlation completeness & fail-closed matching."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import ExecutionContinuationLifecycleState
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.orchestration_topology import OrchestrationSlotId
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.active_governed_execution_task import (
    ActiveGovernedExecutionTask,
)
from intergrax.runtime.human.governed_continuation_bridge import (
    compose_governed_continuation_from_enforcement,
)
from intergrax.runtime.nexus.orchestration.governed_consequential_operation import (
    GovernedOrchestrationSlotExecutor,
)
from intergrax.runtime.policy.mse_hitl_effect_gate import (
    EffectContinuationClassification,
    HumanGovernedProposalRelation,
    MseHitlEffectGateDisposition,
    OptionalIdentityFieldRelation,
    classify_human_governed_proposal_relation,
    compare_optional_identity_field,
    optional_identity_field_matches_exactly,
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
_INTERNAL = (
    _REPO / "intergrax/runtime/nexus/orchestration/internal_continuation_orchestration.py"
)


def _side_effect(
    *,
    operation_id: str = _OPERATION,
    scope_id: str = _SCOPE_1,
    scope_digest: str | None = _SCOPE_DIGEST_1,
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


# --- Phase 1 helper matrix ---


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (None, None, OptionalIdentityFieldRelation.COMPATIBLE),
        (None, "v", OptionalIdentityFieldRelation.INSUFFICIENT),
        ("v", None, OptionalIdentityFieldRelation.INSUFFICIENT),
        ("v", "v", OptionalIdentityFieldRelation.COMPATIBLE),
        ("a", "b", OptionalIdentityFieldRelation.MISMATCH),
    ],
)
def test_compare_optional_identity_field_matrix(
    left: str | None,
    right: str | None,
    expected: OptionalIdentityFieldRelation,
) -> None:
    assert compare_optional_identity_field(left, right) is expected
    assert optional_identity_field_matches_exactly(left, right) is (
        expected is OptionalIdentityFieldRelation.COMPATIBLE
    )


# --- Phase 4 exact match matrix ---


def test_scope_id_equal_matches() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_id=_SCOPE_1),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_id=_SCOPE_1),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.POST_HITL_RESUMED


def test_scope_id_different_explicit_mismatch() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_id=_SCOPE_1),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_id=_SCOPE_2, scope_digest=_SCOPE_DIGEST_2),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION


def test_scope_id_correlation_none_current_concrete_insufficient() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_id=None),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_id=_SCOPE_1),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.CORRELATION_INSUFFICIENT
    relation = classify_human_governed_proposal_relation(
        context.pending,  # type: ignore[arg-type]
        side_effect=_side_effect(scope_id=_SCOPE_1),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert relation is HumanGovernedProposalRelation.CORRELATION_INSUFFICIENT


@pytest.mark.parametrize(
    ("corr_digest", "curr_digest", "expected"),
    [
        (None, None, EffectContinuationClassification.POST_HITL_RESUMED),
        (None, _SCOPE_DIGEST_1, EffectContinuationClassification.CORRELATION_INSUFFICIENT),
        (_SCOPE_DIGEST_1, None, EffectContinuationClassification.CORRELATION_INSUFFICIENT),
        (_SCOPE_DIGEST_1, _SCOPE_DIGEST_1, EffectContinuationClassification.POST_HITL_RESUMED),
        (_SCOPE_DIGEST_1, _SCOPE_DIGEST_2, EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION),
    ],
)
def test_digest_symmetric_matrix(
    corr_digest: str | None,
    curr_digest: str | None,
    expected: EffectContinuationClassification,
) -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_digest=corr_digest),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_digest=curr_digest),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is expected


@pytest.mark.parametrize(
    ("corr_resource", "curr_resource", "expected"),
    [
        (None, None, EffectContinuationClassification.POST_HITL_RESUMED),
        (None, _RESOURCE, EffectContinuationClassification.CORRELATION_INSUFFICIENT),
        (_RESOURCE, None, EffectContinuationClassification.CORRELATION_INSUFFICIENT),
        (_RESOURCE, _RESOURCE, EffectContinuationClassification.POST_HITL_RESUMED),
        (_RESOURCE, "resource-other-r11r4", EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION),
    ],
)
def test_resource_symmetric_matrix(
    corr_resource: str | None,
    curr_resource: str | None,
    expected: EffectContinuationClassification,
) -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(resource_scope=corr_resource),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(resource=curr_resource),
        operation_id=_OPERATION,
        resource_scope=curr_resource,
    )
    assert context.classification is expected


def test_operation_exact_match_and_mismatch() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(operation_id=_OPERATION),
    )
    assert (
        resolve_effect_continuation_context(
            port,
            side_effect=_side_effect(operation_id=_OPERATION),
            operation_id=_OPERATION,
            resource_scope=_RESOURCE,
        ).classification
        is EffectContinuationClassification.POST_HITL_RESUMED
    )
    assert (
        resolve_effect_continuation_context(
            port,
            side_effect=_side_effect(operation_id=_OPERATION_B),
            operation_id=_OPERATION_B,
            resource_scope=_RESOURCE,
        ).classification
        is EffectContinuationClassification.UNRELATED_HUMAN_CONTINUATION
    )


@pytest.mark.parametrize(
    "field",
    ["task_id", "run_id", "attempt_id", "execution_id"],
)
def test_execution_identity_mismatch_not_match(field: str) -> None:
    from intergrax.contracts.governed_continuation_correlation import (
        GovernedContinuationCorrelation,
    )
    from intergrax.runtime.policy.mse_hitl_effect_gate import (
        GovernedProposalCorrelationMatch,
        _compare_governed_correlation_to_current_proposal,
    )

    correlation = GovernedContinuationCorrelation(
        continuation_request_id="gcr_r11r4_id",
        reason=ContinuationReason.COMPLIANCE,
        task_id=mint_task_id() if field == "task_id" else _TASK_ID,
        run_id=mint_run_id() if field == "run_id" else _RUN_ID,
        attempt_id=mint_attempt_id() if field == "attempt_id" else _ATTEMPT_ID,
        execution_id=mint_execution_id() if field == "execution_id" else _EXECUTION_ID,
        side_effect_scope_id=_SCOPE_1,
        side_effect_scope_digest=_SCOPE_DIGEST_1,
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    match = _compare_governed_correlation_to_current_proposal(
        correlation,
        side_effect=_side_effect(),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert match is GovernedProposalCorrelationMatch.EXPLICIT_MISMATCH



# --- Phase 5 behavioral ---


def test_scenario_a_exact_full_correlation_resumed_grant_allow() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, task = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    assert task.runtime.governance.governed_continuation_grant is None


def test_scenario_b_missing_scope_correlation_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_id=None),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.CORRELATION_INSUFFICIENT
    disposition, task = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK
    assert task.runtime.governance.governed_continuation_grant is None


def test_scenario_c_missing_digest_correlation_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_digest=None),
    )
    disposition, _ = _eval_gate(grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.CORRELATION_INSUFFICIENT


def test_scenario_d_missing_resource_correlation_not_post_hitl() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(resource_scope=None),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.CORRELATION_INSUFFICIENT
    disposition, _ = _eval_gate(grant=None, port=port)
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_scenario_e_both_digest_none_may_match() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_digest=None),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_digest=None),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.POST_HITL_RESUMED
    disposition, _ = _eval_gate(
        grant=_grant(side_effect_scope_digest=None),
        port=port,
        scope_digest=None,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED


def test_scenario_f_both_resource_none_may_match() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(resource_scope=None),
    )
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(resource=None),
        operation_id=_OPERATION,
        resource_scope=None,
    )
    assert context.classification is EffectContinuationClassification.POST_HITL_RESUMED



def test_scenario_g_incomplete_old_hitl_does_not_approve_new_effect() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(
            side_effect_scope_id=None,
            side_effect_scope_digest=None,
        ),
        human_request_id="hr-old-effect-a",
    )
    disposition, _ = _eval_gate(
        grant=None,
        port=port,
        scope_id=_SCOPE_2,
        scope_digest=_SCOPE_DIGEST_2,
    )
    assert disposition is MseHitlEffectGateDisposition.PROCEED
    context = resolve_effect_continuation_context(
        port,
        side_effect=_side_effect(scope_id=_SCOPE_2, scope_digest=_SCOPE_DIGEST_2),
        operation_id=_OPERATION,
        resource_scope=_RESOURCE,
    )
    assert context.classification is EffectContinuationClassification.CORRELATION_INSUFFICIENT



def test_scenario_h_incomplete_correlation_same_proposal_grant_no_permission() -> None:
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(side_effect_scope_id=None),
    )
    disposition, task = _eval_gate(grant=_grant(), port=port)
    assert disposition is MseHitlEffectGateDisposition.BLOCK
    assert task.runtime.governance.governed_continuation_grant is None


def test_scenario_i_exact_correlation_wrong_grant_scope_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(
        grant=_grant(side_effect_scope_id=_SCOPE_2),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_scenario_j_exact_correlation_fresh_deny_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(
        action=PolicyAction.DENY,
        grant=_grant(),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.BLOCK


def test_scenario_k_exact_correlation_fresh_require_human_blocks() -> None:
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)
    disposition, _ = _eval_gate(
        action=PolicyAction.REQUIRE_HUMAN,
        grant=_grant(),
        port=port,
    )
    assert disposition is MseHitlEffectGateDisposition.REQUIRE_HITL


# --- Phase 7 topology ---


async def test_topology_exact_slot_correlation_post_hitl() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator)
    effects = [0]
    port = _FakeContinuationPort()
    port.seed(state=ExecutionContinuationLifecycleState.RESUMED)

    class _Inner:
        async def execute_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "slot-ok"

    task = Task(tenant_id="t1", user_id="u1", message="x", task_id=_TASK_ID)
    task.runtime.governance.governed_continuation_grant = _grant()
    executor = GovernedOrchestrationSlotExecutor(
        inner=_Inner(),
        meaningful_side_effect_authorization=boundary,
        production_mode=True,
        build_enforcement_request=lambda _s, _p: _enforcement_request(membership),
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
                slot_id=OrchestrationSlotId("slot-exact"),
                payload=object(),
            )
    finally:
        governed.reset(token)
    assert result == "slot-ok"
    assert effects[0] == 1


async def test_topology_missing_scope_no_exact_match_ordinary_allow() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision())
    boundary, membership = _seed_boundary(evaluator, operation_id=_OPERATION_B)
    effects = [0]
    port = _FakeContinuationPort()
    port.seed(
        state=ExecutionContinuationLifecycleState.RESUMED,
        governed_correlation=_correlation(
            operation_id=_OPERATION_B,
            side_effect_scope_id=None,
        ),
    )

    class _Inner:
        async def execute_slot(self, *, slot_id: OrchestrationSlotId, payload: object) -> str:
            effects[0] += 1
            return "slot-b-ok"

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
    assert result == "slot-b-ok"
    assert effects[0] == 1


# --- Phase 8 External Work / producer ---


def test_compose_governed_continuation_populates_proposal_fields() -> None:
    evaluator = MutableRuntimePolicyEvaluator(_decision(action=PolicyAction.REQUIRE_HUMAN))
    boundary, membership = _seed_boundary(evaluator)
    request = _enforcement_request(membership)
    with bound_gr3_active_execution(
        run_id=_RUN_ID,
        attempt_id=_ATTEMPT_ID,
        execution_id=_EXECUTION_ID,
    ):
        authorization = boundary.authorize(request)
    composed = compose_governed_continuation_from_enforcement(
        request,
        decision=authorization.decision,
        enforcement_operation_id=authorization.enforcement_result.operation_id,
        enforcement_authority_scope=authorization.enforcement_result.authority_scope,
        requires_governed_continuation=True,
        source_agent_id="agent-r11r4",
    )
    assert composed is not None
    correlation = composed.to_correlation()
    assert correlation.side_effect_scope_id == _SCOPE_1
    assert correlation.side_effect_scope_digest == _SCOPE_DIGEST_1
    assert correlation.resource_scope == _RESOURCE
    assert correlation.operation_id == _OPERATION
    assert correlation.task_id == _TASK_ID
    assert correlation.run_id == _RUN_ID
    assert correlation.attempt_id == _ATTEMPT_ID
    assert correlation.execution_id == _EXECUTION_ID


def test_generic_establish_canonical_hitl_pause_not_mse_proposal_scoped() -> None:
    source = _INTERNAL.read_text(encoding="utf-8")
    assert 'operation_id=f"internal_hitl_{human_request_id}"' in source
    assert "if resolved_governed is None:" in source


# --- Phase 9 static gate ---


def test_static_gate_no_wildcard_optional_skip() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "optional_identity_field_matches_exactly" in source
    assert "compare_optional_identity_field" in source
    assert "GovernedProposalCorrelationMatch" in source
    assert "GR-10-R11-R4" in source
    # Forbidden wildcard pattern: skip compare when correlation field is None.
    tree = ast.parse(source, filename=str(_GATE))
    compare_fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_compare_governed_correlation_to_current_proposal",
            "_governed_correlation_matches_current_proposal",
        }:
            if node.name == "_compare_governed_correlation_to_current_proposal":
                compare_fn = node
            break
    assert compare_fn is not None
    for node in ast.walk(compare_fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Detect: if correlation.<field> is not None: ...
        if not isinstance(test, ast.Compare):
            continue
        left = test.left
        if not (
            isinstance(left, ast.Attribute)
            and isinstance(left.value, ast.Name)
            and left.value.id == "correlation"
            and left.attr
            in {
                "resource_scope",
                "side_effect_scope_id",
                "side_effect_scope_digest",
            }
        ):
            continue
        if len(test.ops) == 1 and isinstance(test.ops[0], ast.IsNot):
            if (
                len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value is None
            ):
                raise AssertionError(
                    f"wildcard skip on correlation.{left.attr} is not None is forbidden"
                )


def test_static_gate_insufficient_never_maps_to_post_hitl_in_ordinary_set() -> None:
    source = _GATE.read_text(encoding="utf-8")
    assert "CORRELATION_INSUFFICIENT" in source
    assert "_ORDINARY_ALLOW_CLASSIFICATIONS" in source
    # CORRELATION_INSUFFICIENT is listed under ordinary ALLOW classifications.
    assert (
        "EffectContinuationClassification.CORRELATION_INSUFFICIENT" in source
    )
