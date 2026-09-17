# © Artur Czarnecki. All rights reserved.

"""GR-7-A8-R1 — early lifecycle reliability evidence wired at canonical owners."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest

from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    RecordingAdmissionPort,
    _CREATE_IDEMP,
    _PRINCIPAL,
    _PROVIDER,
    _TENANT,
    _UNCERTAIN_IDEMP,
    _WORKSPACE,
    _create_step,
    _policy_bundle,
)
from applications.governed_contractor_application.tests.host.test_gr7_a3_durable_provider_invocation import (
    RecordingProviderInvocationStore,
    _runtime_with_store,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
)
from intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence import (
    ProviderInvocationReliabilityFact,
    ProviderInvocationReliabilityTracePhase,
)
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.external_work import ExternalWorkErrorCode
from intergrax.contracts.provider_invocation import ProviderInvocationStatus
from intergrax.contracts.provider_invocation_store import ProviderInvocationStore
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)


@dataclass
class _RecordingObserver:
    facts: list[ProviderInvocationReliabilityFact] = field(default_factory=list)

    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        self.facts.append(fact)


class _FailingObserver:
    def observe_provider_invocation_reliability_fact(
        self,
        fact: ProviderInvocationReliabilityFact,
    ) -> None:
        raise RuntimeError("sink down")


def _runtime(
    fake: DeterministicExternalWorkFake,
    task_id: object,
    store: ProviderInvocationStore,
    observer: _RecordingObserver | _FailingObserver | None,
):
    execution_store = InMemoryGovernedExecutionStore()
    receipt_store = InMemoryProofReceiptStore()
    bundle_store = InMemoryPolicyBundleArtifactStore()
    continuation_store = InMemoryContinuationStateStore()
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    bridge = GovernedExternalWorkEnterpriseReliabilityBridge(
        admission_port=RecordingAdmissionPort(),
    )
    return build_governed_external_work_production_runtime(
        fake,
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
        task_scope=StaticActiveTaskScope(task_id),  # type: ignore[arg-type]
        capabilities=quote_first_partner_capability_fixture(provider_id=_PROVIDER),
        policy_bundle=_policy_bundle(),  # type: ignore[arg-type]
        collaborative_work_repositories=cw,
        execution_store=execution_store,
        receipt_store=receipt_store,
        bundle_store=bundle_store,
        continuation_store=continuation_store,
        provider_invocation_store=store,
        reliability_bridge=bridge,
        reliability_evidence_observer=observer,
        clock=lambda: _T0,
    )


def _phases(observer: _RecordingObserver) -> list[ProviderInvocationReliabilityTracePhase]:
    return [f.phase for f in observer.facts]


def test_runtime_without_observer_matches_a3_success() -> None:
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.governed_result is not None


def test_happy_path_emits_governance_intent_dispatch_outcome_succeeded() -> None:
    observer = _RecordingObserver()
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime(fake, task_id, store, observer)
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.governed_result is not None
    inv_id = step.governed_result.provider_invocation.invocation_id
    phases = _phases(observer)
    assert ProviderInvocationReliabilityTracePhase.GOVERNANCE_AUTHORIZED in phases
    assert phases.index(
        ProviderInvocationReliabilityTracePhase.GOVERNANCE_AUTHORIZED,
    ) < phases.index(ProviderInvocationReliabilityTracePhase.INTENT_PERSISTED)
    assert phases.index(
        ProviderInvocationReliabilityTracePhase.INTENT_PERSISTED,
    ) < phases.index(ProviderInvocationReliabilityTracePhase.DISPATCH_ATTEMPTED)
    assert phases.index(
        ProviderInvocationReliabilityTracePhase.DISPATCH_ATTEMPTED,
    ) < phases.index(ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTED)
    gov = next(
        f for f in observer.facts
        if f.phase is ProviderInvocationReliabilityTracePhase.GOVERNANCE_AUTHORIZED
    )
    assert gov.correlation.governance_execution_id is not None
    assert gov.correlation.invocation_id == inv_id
    outcome_fact = next(
        f for f in observer.facts
        if f.phase is ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTED
    )
    assert outcome_fact.invocation_status is ProviderInvocationStatus.SUCCEEDED


def test_unknown_emits_unknown_admitted_not_outcome_persisted() -> None:
    observer = _RecordingObserver()
    fake = DeterministicExternalWorkFake(
        fail_create_with_code={
            _UNCERTAIN_IDEMP: ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN,
        },
    )
    store = RecordingProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime(fake, task_id, store, observer)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_UNCERTAIN_IDEMP,
    )
    assert step.external_effect_outcome is not None
    phases = _phases(observer)
    assert step.external_effect_outcome is ExternalEffectOutcome.UNKNOWN
    assert ProviderInvocationReliabilityTracePhase.UNKNOWN_ADMITTED in phases
    assert ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTED not in phases


def test_intent_persistence_failure_emits_failed_zero_dispatch() -> None:
    observer = _RecordingObserver()
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore(fail_intent=True)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime(fake, task_id, store, observer)
    step, fake_after = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake_after.create_calls == 0
    assert step.reason == "provider_invocation_intent_persistence_failed"
    phases = _phases(observer)
    assert ProviderInvocationReliabilityTracePhase.INTENT_PERSISTENCE_FAILED in phases
    fail_fact = next(
        f
        for f in observer.facts
        if f.phase is ProviderInvocationReliabilityTracePhase.INTENT_PERSISTENCE_FAILED
    )
    assert fail_fact.provider_mutation_count == 0
    assert ProviderInvocationReliabilityTracePhase.DISPATCH_ATTEMPTED not in phases
    assert ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTED not in phases


def test_outcome_persistence_failure_no_durable_outcome_fact() -> None:
    observer = _RecordingObserver()
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore(fail_outcome=True)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime(fake, task_id, store, observer)
    _create_step(runtime, fake, task_id, run_id, attempt_id, execution_id)
    phases = _phases(observer)
    assert ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTENCE_FAILED in phases
    assert ProviderInvocationReliabilityTracePhase.OUTCOME_PERSISTED not in phases
    assert ProviderInvocationReliabilityTracePhase.UNKNOWN_ADMITTED not in phases


def test_null_observer_same_lifecycle_semantics() -> None:
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    fake_a = DeterministicExternalWorkFake()
    store_a = RecordingProviderInvocationStore()
    runtime_without = _runtime(fake_a, task_id, store_a, None)
    step, _ = _create_step(
        runtime_without, fake_a, task_id, run_id, attempt_id, execution_id,
    )
    fake_b = DeterministicExternalWorkFake()
    store_b = RecordingProviderInvocationStore()
    runtime_with_null = _runtime(fake_b, task_id, store_b, None)
    step_b, _ = _create_step(
        runtime_with_null,
        fake_b,
        task_id,
        run_id,
        attempt_id,
        execution_id,
    )
    assert step.governed_result is not None
    assert step_b.governed_result is not None
    assert fake_a.create_calls == fake_b.create_calls == 1


def test_observer_failure_does_not_block_success_path() -> None:
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime(fake, task_id, store, _FailingObserver())
    step, _ = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert step.governed_result is not None
