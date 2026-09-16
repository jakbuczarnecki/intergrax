# © Artur Czarnecki. All rights reserved.

"""Domain pack registration and representative pass/violation paths."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.runtime_invariants import RuntimeInvariantStatus
from intergrax.runtime.execution.delegated_execution.invariants.probe import (
    DelegatedProviderInvariantFacts,
)
from intergrax.runtime.execution.delegated_execution.invariants.pack import (
    DelegatedProviderRuntimeInvariantRulePack,
)
from intergrax.runtime.execution.invariants.probe import ExecutionInvariantFacts
from intergrax.runtime.execution.invariants.pack import ExecutionRuntimeInvariantRulePack
from intergrax.runtime.governance.invariants.probe import (
    GovernanceInvariantFacts,
    GovernanceMeaningfulSideEffectBinding,
)
from intergrax.runtime.governance.invariants.pack import GovernanceRuntimeInvariantRulePack
from intergrax.runtime.invariants.evaluation_id import MonotonicRuntimeInvariantEvaluationIdFactory
from intergrax.runtime.invariants.foundation_composition import (
    compose_foundation_runtime_invariant_service,
    foundation_runtime_invariant_rule_packs,
)
from intergrax.runtime.invariants.composition import compose_default_runtime_invariant_runner
from intergrax.runtime.invariants.service import RuntimeInvariantService
from intergrax.runtime.invariants.clock import SystemRuntimeInvariantEvaluationClock

pytestmark = pytest.mark.unit


class _FakeExecutionProbe:
    def __init__(self, facts: ExecutionInvariantFacts) -> None:
        self._facts = facts

    def read_facts(self) -> ExecutionInvariantFacts:
        return self._facts


class _FakeDelegatedProbe:
    def __init__(self, facts: DelegatedProviderInvariantFacts) -> None:
        self._facts = facts

    def read_facts(self) -> DelegatedProviderInvariantFacts:
        return self._facts


class _FakeGovernanceProbe:
    def __init__(self, facts: GovernanceInvariantFacts) -> None:
        self._facts = facts

    def read_facts(self) -> GovernanceInvariantFacts:
        return self._facts


def _service_from_packs(*packs) -> RuntimeInvariantService:
    rule_packs = packs
    return RuntimeInvariantService(
        rule_packs=rule_packs,
        clock=SystemRuntimeInvariantEvaluationClock(),
        evaluation_id_factory=MonotonicRuntimeInvariantEvaluationIdFactory(),
        runner=compose_default_runtime_invariant_runner(rule_packs),
    )


def test_foundation_packs_stable_rule_ids() -> None:
    packs = foundation_runtime_invariant_rule_packs()
    rule_ids = [rule.rule_id for pack in packs for rule in pack.rules]
    assert "EE-INV-001" in rule_ids
    assert "DELEGATION-INV-001" in rule_ids
    assert "GOV-INV-001" in rule_ids


def test_execution_pack_happy_path_pass() -> None:
    from intergrax.contracts.execution_identity_authority import (
        CANONICAL_IDENTITY_AUTHORITY_MODULE,
        CANONICAL_LIFECYCLE_OWNER_MODULE,
    )

    probe = _FakeExecutionProbe(
        ExecutionInvariantFacts(
            identity_authority_module=CANONICAL_IDENTITY_AUTHORITY_MODULE,
            lifecycle_owner_module=CANONICAL_LIFECYCLE_OWNER_MODULE,
            supported_execution_bypass_active=False,
        ),
    )
    report = _service_from_packs(ExecutionRuntimeInvariantRulePack(probe)).evaluate()
    assert all(r.status is RuntimeInvariantStatus.PASS for r in report.results)


def test_execution_pack_bypass_violation() -> None:
    from intergrax.contracts.execution_identity_authority import (
        CANONICAL_IDENTITY_AUTHORITY_MODULE,
        CANONICAL_LIFECYCLE_OWNER_MODULE,
    )

    probe = _FakeExecutionProbe(
        ExecutionInvariantFacts(
            identity_authority_module=CANONICAL_IDENTITY_AUTHORITY_MODULE,
            lifecycle_owner_module=CANONICAL_LIFECYCLE_OWNER_MODULE,
            supported_execution_bypass_active=True,
        ),
    )
    report = _service_from_packs(ExecutionRuntimeInvariantRulePack(probe)).evaluate()
    assert any(
        r.rule_id == "EE-INV-003" and r.status is RuntimeInvariantStatus.VIOLATION
        for r in report.results
    )


def test_delegated_pack_provider_identity_violation() -> None:
    probe = _FakeDelegatedProbe(
        DelegatedProviderInvariantFacts(provider_claims_canonical_execution_identity=True),
    )
    report = _service_from_packs(DelegatedProviderRuntimeInvariantRulePack(probe)).evaluate()
    assert any(
        r.rule_id == "DELEGATION-INV-001" and r.status is RuntimeInvariantStatus.VIOLATION
        for r in report.results
    )


def test_delegated_pack_correlation_pass() -> None:
    execution_id = mint_execution_id()
    probe = _FakeDelegatedProbe(
        DelegatedProviderInvariantFacts(
            correlation_execution_id=execution_id,
            correlation_binding_execution_id=execution_id,
        ),
    )
    report = _service_from_packs(DelegatedProviderRuntimeInvariantRulePack(probe)).evaluate()
    assert any(
        r.rule_id == "DELEGATION-INV-002" and r.status is RuntimeInvariantStatus.PASS
        for r in report.results
    )


def test_governance_pack_binding_violation() -> None:
    task = mint_task_id()
    run = mint_run_id()
    attempt = mint_attempt_id()
    execution = mint_execution_id()
    probe = _FakeGovernanceProbe(
        GovernanceInvariantFacts(
            meaningful_side_effect_binding=GovernanceMeaningfulSideEffectBinding(
                request_task_id=task,
                request_run_id=run,
                request_attempt_id=attempt,
                request_execution_id=execution,
                active_task_id=task,
                active_run_id=run,
                active_attempt_id=attempt,
                active_execution_id=mint_execution_id(),
            ),
        ),
    )
    report = _service_from_packs(GovernanceRuntimeInvariantRulePack(probe)).evaluate()
    assert any(
        r.rule_id == "GOV-INV-001" and r.status is RuntimeInvariantStatus.VIOLATION
        for r in report.results
    )


def test_foundation_qualification_service_no_unexpected_violations() -> None:
    report = compose_foundation_runtime_invariant_service().evaluate()
    assert report.summary.violation_count == 0
    assert report.summary.evaluation_error_count == 0
