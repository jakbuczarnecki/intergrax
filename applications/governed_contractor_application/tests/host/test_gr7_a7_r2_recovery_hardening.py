# © Artur Czarnecki. All rights reserved.

"""GR-7-A7-R2 — typed metadata, canonical operation identity, repeat capability gating."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from applications.governed_contractor_application.host.governed_external_work_recovery_ports import (
    GovernedExternalWorkProviderRecoveryRepeatPort,
)
from applications.governed_contractor_application.host.provider_invocation_recovery import (
    GovernedExternalWorkProviderRecovery,
)
from applications.governed_contractor_application.host.provider_invocation_reconciliation import (
    GovernedExternalWorkProviderReconciliation,
)
from applications.governed_contractor_application.tests.host.test_gr7_a7_provider_recovery import (
    _AllowRepeatPolicy,
    _RecordingRepeatPort,
    _invocation,
    _outcome,
    _repeat_eligibility,
)
from external_contractor_adapter.external_effect_contracts import (
    external_work_effect_contract_for_action,
)
from external_contractor_adapter.side_effect_actions import (
    ACTION_ACCEPT_QUOTE,
    ACTION_CANCEL_EXTERNAL_WORK,
    ACTION_CREATE_EXTERNAL_WORK,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryAction,
    ProviderInvocationRecoveryDecision,
    ProviderInvocationRecoveryReason,
    ProviderInvocationRecoveryRequest,
    evaluate_provider_invocation_recovery,
)
from intergrax.contracts.external_work import ExternalWorkCreateRequest
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.governed_execution_result import (
    external_work_decision_action_for_provider_operation,
    external_work_invocation_operation_matches_effect_contract,
    external_work_provider_operation_for_decision_action,
    external_work_provider_operation_for_effect_contract_operation_key,
)
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.runtime.enterprise_reliability.provider_invocation_recovery import (
    ProviderInvocationRecoveryExecutionBlockReason,
    ProviderInvocationRecoveryExecutionDisposition,
    ProviderInvocationRecoveryExecutionPorts,
    decide_provider_invocation_recovery,
    execute_provider_invocation_recovery,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_R2_PRODUCTION_FILES = (
    _REPO_ROOT
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "governed_external_work_recovery_ports.py",
    _REPO_ROOT
    / "applications"
    / "governed_contractor_application"
    / "host"
    / "provider_invocation_recovery.py",
)

_T0 = datetime(2026, 9, 17, 14, 0, 0, tzinfo=UTC)
_CAPABILITIES = quote_first_partner_capability_fixture()
_CREATE_CONTRACT = external_work_effect_contract_for_action(
    ACTION_CREATE_EXTERNAL_WORK,
    _CAPABILITIES,
)
_ACCEPT_CONTRACT = external_work_effect_contract_for_action(
    ACTION_ACCEPT_QUOTE,
    _CAPABILITIES,
)
_CANCEL_CONTRACT = external_work_effect_contract_for_action(
    ACTION_CANCEL_EXTERNAL_WORK,
    _CAPABILITIES,
)

_FORBIDDEN_AST_NAMES = frozenset({"Any", "object"})
_FORBIDDEN_SUBSCRIPTS = frozenset(
    {
        ("Mapping", "str", "Any"),
        ("dict", "str", "Any"),
    },
)


def _forbidden_nodes_in_source(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_AST_NAMES:
            violations.append(node.id)
        if isinstance(node, ast.Subscript):
            slice_node = node.slice
            if isinstance(slice_node, ast.Tuple):
                parts: list[str] = []
                for element in slice_node.elts:
                    if isinstance(element, ast.Name):
                        parts.append(element.id)
                if len(parts) == 2 and (parts[0], parts[1]) == ("str", "Any"):
                    base = node.value.id if isinstance(node.value, ast.Name) else ""
                    violations.append(f"{base}[str, Any]")
    if "# type: ignore" in source:
        violations.append("type: ignore")
    return violations


@pytest.mark.parametrize("path", _R2_PRODUCTION_FILES, ids=lambda p: p.name)
def test_r2_production_files_have_no_loose_typing(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    violations = _forbidden_nodes_in_source(source)
    assert violations == []


@pytest.mark.parametrize(
    ("action", "provider_operation"),
    (
        (ACTION_CREATE_EXTERNAL_WORK, "create_work"),
        (ACTION_ACCEPT_QUOTE, "submit_quote_acceptance"),
        (ACTION_CANCEL_EXTERNAL_WORK, "cancel_work"),
    ),
)
def test_canonical_decision_action_provider_operation_binding(
    action: str,
    provider_operation: str,
) -> None:
    assert external_work_provider_operation_for_decision_action(action) == provider_operation
    assert external_work_decision_action_for_provider_operation(provider_operation) == action


def test_legacy_provider_operation_alias_maps_to_create_action() -> None:
    assert (
        external_work_decision_action_for_provider_operation("create_work")
        == ACTION_CREATE_EXTERNAL_WORK
    )


def test_effect_contract_operation_key_matches_provider_invocation() -> None:
    contract = _CREATE_CONTRACT
    assert external_work_invocation_operation_matches_effect_contract(
        contract_operation_key=contract.operation_key,
        invocation_operation="create_work",
    )
    assert external_work_invocation_operation_matches_effect_contract(
        contract_operation_key=contract.operation_key,
        invocation_operation=contract.operation_key,
    )


def test_effect_contract_invocation_mismatch_fail_closed() -> None:
    assert not external_work_invocation_operation_matches_effect_contract(
        contract_operation_key=_ACCEPT_CONTRACT.operation_key,
        invocation_operation="create_work",
    )


def test_host_contract_for_does_not_mutate_operation_key() -> None:
    stack = GovernedExternalWorkProviderRecovery.build(
        GovernedExternalWorkProviderReconciliation.build(DeterministicExternalWorkFake()),
    )
    inv = _invocation().model_copy(update={"operation": "create_work"})
    before = external_work_effect_contract_for_action(ACTION_CREATE_EXTERNAL_WORK, _CAPABILITIES)
    request = stack.build_recovery_request(
        invocation=inv,
        outcome=_outcome(inv.invocation_id, ProviderInvocationStatus.FAILED),
        capabilities=_CAPABILITIES,
    )
    after = request.effect_contract
    assert after.operation_key == before.operation_key
    assert after.contract_id == before.contract_id
    assert after.safety == before.safety


@dataclass(frozen=True, slots=True)
class _SelectRepeatPolicy:
    def decide(self, request: object) -> object:
        from intergrax.contracts.enterprise_reliability.provider_invocation_recovery import (
            ProviderInvocationRecoveryPolicyDecision,
            ProviderInvocationRecoveryPolicyRequest,
        )

        assert isinstance(request, ProviderInvocationRecoveryPolicyRequest)
        if ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT in request.allowed_actions:
            return ProviderInvocationRecoveryPolicyDecision(
                selected_action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
            )
        return ProviderInvocationRecoveryPolicyDecision(
            selected_action=request.allowed_actions[-1],
        )

    @property
    def policy_id(self) -> str:
        return "test_select_repeat_r2"


def test_create_repeat_selectable_when_execution_supported() -> None:
    inv = _invocation()
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_CREATE_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
        repeat_execution_supported=True,
    )
    decision = decide_provider_invocation_recovery(request, policy=_SelectRepeatPolicy())
    assert decision.action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT


def test_accept_eligible_but_execution_unsupported_no_repeat() -> None:
    from intergrax.contracts.enterprise_reliability.repeat_eligibility import (
        ExternalEffectRepeatEligibilityRequest,
        ExternalEffectRepeatEligibilityVerdict,
        evaluate_external_effect_repeat_eligibility,
    )

    inv = _invocation().model_copy(update={"operation": "submit_quote_acceptance"})
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    eligibility = evaluate_external_effect_repeat_eligibility(
        ExternalEffectRepeatEligibilityRequest(
            invocation=inv,
            outcome=outcome,
            effect_contract=_ACCEPT_CONTRACT,
        ),
        policy=_AllowRepeatPolicy(),
    )
    assert eligibility.verdict is ExternalEffectRepeatEligibilityVerdict.ELIGIBLE
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_ACCEPT_CONTRACT,
        repeat_eligibility=eligibility,
        repeat_execution_supported=False,
    )
    decision = evaluate_provider_invocation_recovery(request, policy=_SelectRepeatPolicy())
    assert decision.action is not ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT


def test_forged_repeat_on_unsupported_operation_blocked_without_port_call() -> None:
    inv = _invocation().model_copy(update={"operation": "submit_quote_acceptance"})
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    request = ProviderInvocationRecoveryRequest(
        invocation=inv,
        outcome=outcome,
        effect_contract=_ACCEPT_CONTRACT,
        repeat_eligibility=_repeat_eligibility(inv, outcome),
        repeat_execution_supported=False,
    )
    decision = ProviderInvocationRecoveryDecision(
        action=ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT,
        reason=ProviderInvocationRecoveryReason.POLICY_SELECTED,
        invocation_id=inv.invocation_id,
        idempotency_key=inv.idempotency_key,
    )
    repeat_port = _RecordingRepeatPort()
    result = execute_provider_invocation_recovery(
        request,
        decision=decision,
        ports=ProviderInvocationRecoveryExecutionPorts(repeat=repeat_port),
    )
    assert repeat_port.calls == 0
    assert result.provider_mutation_count == 0
    assert result.disposition is ProviderInvocationRecoveryExecutionDisposition.BLOCKED
    assert (
        result.block_reason
        is ProviderInvocationRecoveryExecutionBlockReason.REPEAT_EXECUTION_UNSUPPORTED
    )


def test_governed_repeat_port_supports_create_only() -> None:
    from external_contractor_adapter.external_work_adapter import ExternalWorkAdapter
    from governed_contractor_application.host.stores import InMemoryProviderInvocationStore

    fake = DeterministicExternalWorkFake()
    port = GovernedExternalWorkProviderRecoveryRepeatPort(
        adapter=ExternalWorkAdapter(fake),
        invocation_store=InMemoryProviderInvocationStore(),
        principal_id="principal-r2",
        tenant_id="tenant-r2",
        run_id="run-r2",
        attempt_id="attempt-r2",
        execution_id="execution-r2",
        create_request_for_invocation=lambda inv: ExternalWorkCreateRequest(
            provider_id=inv.provider_id,
            task_id=inv.task_id,
            run_id=inv.run_id,
            scope_description="scope",
            scope_digest=inv.request_digest,
            idempotency_key=inv.idempotency_key or "idem",
        ),
        clock=lambda: _T0,
        host_execution_id="exec-r2",
    )
    assert port.supports_provider_operation("create_work")
    assert port.supports_provider_operation("external_work.create_work")
    assert not port.supports_provider_operation("submit_quote_acceptance")


@dataclass
class _CustomAcceptRepeatPort:
    def supports_provider_operation(self, operation: str) -> bool:
        return operation == "submit_quote_acceptance"

    def execute_idempotent_repeat(
        self,
        *,
        original_invocation: object,
        original_outcome: object,
        effect_contract: object,
    ) -> object:
        _ = original_invocation, original_outcome, effect_contract
        raise AssertionError("must not be called in decision-only test")


def test_custom_repeat_port_can_enable_accept_without_erl_core_change() -> None:
    inv = _invocation().model_copy(update={"operation": "submit_quote_acceptance"})
    outcome = _outcome(inv.invocation_id, ProviderInvocationStatus.FAILED)
    stack = GovernedExternalWorkProviderRecovery.build(
        GovernedExternalWorkProviderReconciliation.build(DeterministicExternalWorkFake()),
    )
    custom = _CustomAcceptRepeatPort()
    request = stack.build_recovery_request(
        invocation=inv,
        outcome=outcome,
        capabilities=_CAPABILITIES,
        repeat_policy=_AllowRepeatPolicy(),
        repeat_port=custom,
    )
    assert request.repeat_execution_supported is True
    decision = decide_provider_invocation_recovery(request, policy=_SelectRepeatPolicy())
    assert decision.action is ProviderInvocationRecoveryAction.IDEMPOTENT_REPEAT


def test_effect_contract_operation_key_maps_to_provider_operation() -> None:
    assert (
        external_work_provider_operation_for_effect_contract_operation_key(
            _CREATE_CONTRACT.operation_key,
        )
        == "create_work"
    )
    assert (
        external_work_provider_operation_for_effect_contract_operation_key(
            _ACCEPT_CONTRACT.operation_key,
        )
        == "submit_quote_acceptance"
    )
    assert (
        external_work_provider_operation_for_effect_contract_operation_key(
            _CANCEL_CONTRACT.operation_key,
        )
        == "cancel_work"
    )