# © Artur Czarnecki. All rights reserved.

"""PG-FIX-A — External Work canonical side-effect authorization boundary convergence."""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from external_contractor_adapter.external_work_adapter import (
    META_IDEMPOTENCY_KEY,
    META_PROVIDER_ID,
    META_SCOPE_DESCRIPTION,
    META_SCOPE_DIGEST,
    META_WORKSPACE_REF,
    ExternalWorkAdapter,
    adapt_from_step_metadata,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from external_contractor_adapter.tests.fakes.deterministic_side_effect_policy import (
    DeterministicMeaningfulSideEffectPolicy,
)
from external_contractor_adapter.tests.fakes.external_work_authorization_boundary import (
    allow_external_work_boundary,
    seed_external_work_authorization_boundary,
)
from intergrax.contracts.actor_identity import ActorIdentity, ActorKind
from intergrax.contracts.collaborative_work import (
    CollaborativeWorkEnforcementRequest,
    MembershipResolutionMode,
)
from intergrax.contracts.external_work import QuoteAcceptanceEvidence
from intergrax.contracts.money import MoneyAmount
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationResult,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + ("cd" * 32)
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_TASK = "task-pg-a1"
_RUN = "run-pg-a1"
_PRINCIPAL = "principal-pg-a1"
_ADAPTER_PY = Path(__file__).resolve().parents[4] / "agents" / "external_contractor_adapter" / "external_work_adapter.py"


class _RecordingIntegration(DeterministicExternalWorkFake):
    def __init__(self, *, call_log: list[str] | None = None) -> None:
        super().__init__()
        self.call_log = call_log if call_log is not None else []

    def create_work(self, request):  # type: ignore[no-untyped-def]
        self.call_log.append("integration.create_work")
        return super().create_work(request)

    def submit_quote_acceptance(self, correlation, acceptance, *, idempotency_key: str):
        self.call_log.append("integration.accept_quote")
        return super().submit_quote_acceptance(
            correlation, acceptance, idempotency_key=idempotency_key
        )


class _SpyAuthorizationBoundary:
    def __init__(
        self,
        *,
        result: MeaningfulSideEffectAuthorizationResult | None = None,
        raise_on_authorize: Exception | None = None,
    ) -> None:
        self.calls: list[CollaborativeWorkEnforcementRequest] = []
        self._result = result
        self._raise = raise_on_authorize

    def authorize(
        self,
        request: CollaborativeWorkEnforcementRequest,
        *,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
    ) -> MeaningfulSideEffectAuthorizationResult:
        _ = source_step_id
        self.calls.append(request)
        if self._raise is not None:
            raise self._raise
        if self._result is not None:
            return self._result
        decision = PolicyDecision(action=PolicyAction.ALLOW, reason="spy-allow")
        enforcement = MagicMock(operation_id=request.operation_id, authority_scope="spy")
        return MeaningfulSideEffectAuthorizationResult(
            permitted=True,
            decision=decision,
            enforcement_result=enforcement,
            requires_governed_continuation=False,
        )

    def authorize_and_execute(
        self,
        request: CollaborativeWorkEnforcementRequest,
        execute: object,
        *,
        task: object | None = None,
        lifecycle: object | None = None,
        source_agent_id: str = "platform.meaningful_side_effect",
        source_step_id: str | None = None,
        on_authorization: object | None = None,
    ) -> object:
        _ = task, lifecycle, source_step_id
        authorization = self.authorize(
            request,
            source_agent_id=source_agent_id,
        )
        if on_authorization is not None:
            on_authorization(authorization)
        if authorization.permitted:
            return execute()
        return authorization


def _meta(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        META_PROVIDER_ID: "gec3_deterministic_fake",
        META_SCOPE_DESCRIPTION: "review PR #42",
        META_SCOPE_DIGEST: _DIGEST,
        META_IDEMPOTENCY_KEY: "idem-pg-fix-a",
        META_WORKSPACE_REF: _WORKSPACE,
        "external_work.budget_limit": MoneyAmount(amount=Decimal("40.00"), currency="USD"),
        "external_work.principal_id": _PRINCIPAL,
        "external_work.tenant_id": _TENANT,
    }
    payload.update(overrides)
    return payload


def _acceptance() -> QuoteAcceptanceEvidence:
    return QuoteAcceptanceEvidence(
        acceptance_id="acc-pg-fix-a",
        quote_id="q-pg-fix-a",
        quote_version=1,
        scope_digest=_DIGEST,
        actor=ActorIdentity(kind=ActorKind.USER, actor_id=_PRINCIPAL, tenant_id=_TENANT),
        accepted_at=datetime(2026, 8, 18, 12, 0, tzinfo=UTC),
        hitl_decision_id="hdec-pg-fix-a",
    )


def _deny_result(*, reason: str = "deny") -> MeaningfulSideEffectAuthorizationResult:
    decision = PolicyDecision(action=PolicyAction.DENY, reason=reason, policy_rule_id="spy.deny")
    enforcement = MagicMock(operation_id=ACTION_CREATE_EXTERNAL_WORK, authority_scope="spy")
    return MeaningfulSideEffectAuthorizationResult(
        permitted=False,
        decision=decision,
        enforcement_result=enforcement,
        requires_governed_continuation=False,
    )


def test_a1_external_work_calls_canonical_boundary_before_provider() -> None:
    call_log: list[str] = []
    spy = _SpyAuthorizationBoundary()
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=spy)
    result = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert result.used is True
    assert len(spy.calls) == 1
    assert spy.calls[0].operation_id == ACTION_CREATE_EXTERNAL_WORK
    assert "integration.create_work" in call_log


def test_a2_no_parallel_evaluator_in_production_adapter() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_ADAPTER_PY))
    assert "MeaningfulSideEffectEvaluator" not in source
    assert "evaluate_meaningful_side_effect(" not in source
    assert "WorkspaceMembership(" not in source
    assert "MembershipStatus.ACTIVE" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "evaluate_meaningful_side_effect":
            pytest.fail("adapter must not call evaluate_meaningful_side_effect directly")


def test_a3_missing_boundary_fails_closed() -> None:
    call_log: list[str] = []
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=None)
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert denied.used is False
    assert denied.reason == "side_effect_authorization_boundary_missing"
    assert call_log == []


def test_a4_allow_executes_provider_once() -> None:
    call_log: list[str] = []
    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    boundary = allow_external_work_boundary(
        runtime_policy_evaluator=runtime,
        principal_id=_PRINCIPAL,
    )
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=boundary)
    result = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert result.used is True
    assert call_log.count("integration.create_work") == 1


def test_a5_deny_blocks_provider() -> None:
    call_log: list[str] = []
    spy = _SpyAuthorizationBoundary(result=_deny_result())
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=spy)
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert call_log == []


def test_a6_require_human_without_grant_blocks_provider() -> None:
    call_log: list[str] = []
    runtime = DeterministicMeaningfulSideEffectPolicy(
        default=PolicyAction.ALLOW,
        by_action={ACTION_ACCEPT_QUOTE: PolicyAction.REQUIRE_HUMAN},
    )
    boundary = allow_external_work_boundary(
        runtime_policy_evaluator=runtime,
        principal_id=_PRINCIPAL,
    )
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=boundary)
    created = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        enrich=False,
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert created.snapshot is not None and created.quote is not None
    call_log.clear()
    gated = adapter.forward_quote_acceptance(
        created.snapshot.correlation,
        _acceptance(),
        idempotency_key="idem-accept-pg",
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
    )
    assert gated.used is False
    assert gated.reason == "side_effect_governance_required"
    assert call_log == []


def test_a7_matching_grant_dependency_pg_fix_c() -> None:
    source = _ADAPTER_PY.read_text(encoding="utf-8")
    assert "authorize_and_execute" in source
    assert "authorization_boundary.authorize(" not in source


def test_a8_deny_overrides_grant_blocks_provider() -> None:
    call_log: list[str] = []
    spy = _SpyAuthorizationBoundary(result=_deny_result(reason="fresh-deny"))
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=spy)
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert denied.used is False
    assert denied.policy_decision is not None
    assert denied.policy_decision.action is PolicyAction.DENY
    assert call_log == []


def test_a9_tenant_isolation_uses_real_gate() -> None:
    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    boundary = seed_external_work_authorization_boundary(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        runtime_policy_evaluator=runtime,
    )
    adapter = ExternalWorkAdapter(DeterministicExternalWorkFake(), authorization_boundary=boundary)
    denied = adapter.create_and_map(
        adapter.build_create_request(
            task_id=_TASK,
            run_id=_RUN,
            metadata=_meta(**{"external_work.tenant_id": "tenant-b"}),
        ),
        principal_id=_PRINCIPAL,
        tenant_id="tenant-b",
    )
    assert denied.used is False
    assert denied.reason == "side_effect_denied"


def test_a10_principal_without_authority_denied() -> None:
    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    boundary = seed_external_work_authorization_boundary(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        runtime_policy_evaluator=runtime,
    )
    adapter = ExternalWorkAdapter(DeterministicExternalWorkFake(), authorization_boundary=boundary)
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id="unauthorized-principal",
        tenant_id=_TENANT,
    )
    assert denied.used is False
    assert denied.reason == "side_effect_denied"


def test_a11_resource_scope_changes_authorization() -> None:
    other_digest = "sha256:" + ("ef" * 32)
    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    boundary = seed_external_work_authorization_boundary(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        runtime_policy_evaluator=runtime,
        seed_resource_policy=True,
        resource_deny_scopes=(_DIGEST,),
        resource_allow_scopes=(other_digest,),
    )
    adapter = ExternalWorkAdapter(DeterministicExternalWorkFake(), authorization_boundary=boundary)
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert denied.used is False
    allowed = adapter.create_and_map(
        adapter.build_create_request(
            task_id="task-other",
            run_id="run-other",
            metadata=_meta(**{META_SCOPE_DIGEST: other_digest, META_IDEMPOTENCY_KEY: "idem-other"}),
        ),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert allowed.used is True


def test_a12_same_action_different_authority() -> None:
    call_log_allowed: list[str] = []
    call_log_denied: list[str] = []
    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    boundary = seed_external_work_authorization_boundary(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        runtime_policy_evaluator=runtime,
    )
    adapter_allowed = ExternalWorkAdapter(
        _RecordingIntegration(call_log=call_log_allowed),
        authorization_boundary=boundary,
    )
    adapter_denied = ExternalWorkAdapter(
        _RecordingIntegration(call_log=call_log_denied),
        authorization_boundary=boundary,
    )
    allowed = adapter_allowed.create_and_map(
        adapter_allowed.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    denied = adapter_denied.create_and_map(
        adapter_denied.build_create_request(
            task_id="task-no-membership",
            run_id="run-no-membership",
            metadata=_meta(**{META_IDEMPOTENCY_KEY: "idem-no-membership"}),
        ),
        principal_id="principal-without-membership",
        tenant_id=_TENANT,
    )
    assert allowed.used is True
    assert call_log_allowed.count("integration.create_work") == 1
    assert denied.used is False
    assert denied.reason == "side_effect_denied"
    assert call_log_denied == []


def test_a13_observational_get_quote_skips_authorization() -> None:
    spy = _SpyAuthorizationBoundary()
    fake = DeterministicExternalWorkFake()
    adapter = ExternalWorkAdapter(fake, authorization_boundary=spy)
    snapshot = fake.create_work(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta())
    )
    spy.calls.clear()
    result = adapter.map_existing(snapshot.correlation)
    assert result.used is True
    assert spy.calls == []


def test_a14_policy_fault_fails_closed() -> None:
    call_log: list[str] = []
    spy = _SpyAuthorizationBoundary(raise_on_authorize=RuntimeError("gate fault"))
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=spy)
    denied = adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert denied.used is False
    assert denied.reason == "side_effect_authorization_failed"
    assert call_log == []


def test_a15_identity_dimensions_preserved_in_enforcement_request() -> None:
    spy = _SpyAuthorizationBoundary()
    adapter = ExternalWorkAdapter(DeterministicExternalWorkFake(), authorization_boundary=spy)
    adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    req = spy.calls[0]
    side_effect = req.meaningful_side_effect_request
    assert side_effect is not None
    assert side_effect.task_id == _TASK
    assert side_effect.run_id == _RUN
    assert side_effect.task_id != side_effect.run_id
    assert side_effect.principal_id == _PRINCIPAL
    assert side_effect.tenant_id == _TENANT
    assert req.tenant_id == _TENANT
    assert req.workspace_id == _WORKSPACE
    assert req.acting_principal_id == _PRINCIPAL
    assert req.membership is None
    assert req.membership_resolution_mode is MembershipResolutionMode.CANONICAL_PRINCIPAL


def test_a16_authorization_before_provider_ordering() -> None:
    call_log: list[str] = []
    events: list[str] = []

    class _OrderedSpy(_SpyAuthorizationBoundary):
        def authorize(self, request, **kwargs):  # type: ignore[no-untyped-def]
            events.append("boundary.authorize")
            return super().authorize(request, **kwargs)

    spy = _OrderedSpy()
    adapter = ExternalWorkAdapter(_RecordingIntegration(call_log=call_log), authorization_boundary=spy)
    adapter.create_and_map(
        adapter.build_create_request(task_id=_TASK, run_id=_RUN, metadata=_meta()),
        principal_id=_PRINCIPAL,
        tenant_id=_TENANT,
    )
    assert events == ["boundary.authorize"]
    assert call_log == ["integration.create_work"]


def test_adapt_from_step_metadata_wires_authorization_boundary() -> None:
    runtime = DeterministicMeaningfulSideEffectPolicy(default=PolicyAction.ALLOW)
    boundary = allow_external_work_boundary(
        runtime_policy_evaluator=runtime,
        principal_id=_PRINCIPAL,
    )
    result = adapt_from_step_metadata(
        DeterministicExternalWorkFake(),
        task_id=_TASK,
        run_id=_RUN,
        message="scope",
        metadata=_meta(),
        authorization_boundary=boundary,
    )
    assert result.used is True
