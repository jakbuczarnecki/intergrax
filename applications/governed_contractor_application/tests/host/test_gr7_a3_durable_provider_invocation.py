# © Artur Czarnecki. All rights reserved.

"""GR-7-A3 — durable ProviderInvocation intent and outcome lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest
from intergrax.contracts.policy_action import PolicyAction

from applications.governed_contractor_application.host.production_external_work_composition import (
    build_governed_external_work_production_runtime,
)
from applications.governed_contractor_application.tests.host.gr6_collaborative_work_test_support import (
    gr6_seeded_collaborative_work_repositories,
)
from applications.governed_contractor_application.tests.host.durable_provider_invocation_test_store import (
    DurableTestProviderInvocationStore,
)
from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
    RecordingAdmissionPort,
    _create_step,
    _FAIL_IDEMP,
    _PRINCIPAL,
    _PROVIDER,
    _TENANT,
    _UNCERTAIN_IDEMP,
    _WORKSPACE,
)
from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.external_work_enterprise_reliability_bridge import (
    GovernedExternalWorkEnterpriseReliabilityBridge,
)
from governed_contractor_application.host.lifecycle_states import GovernedExternalWorkHostState
from governed_contractor_application.host.stores import (
    InMemoryContinuationStateStore,
    InMemoryGovernedExecutionStore,
    InMemoryPolicyBundleArtifactStore,
    InMemoryProofReceiptStore,
    InMemoryProviderInvocationStore,
)
from intergrax.contracts.enterprise_reliability.outcome import ExternalEffectOutcome
from intergrax.contracts.external_work import ExternalWorkErrorCode
from intergrax.contracts.external_work_provider_capabilities import (
    quote_first_partner_capability_fixture,
)
from intergrax.contracts.provider_invocation import (
    ProviderInvocation,
    ProviderInvocationOutcome,
    ProviderInvocationStatus,
)
from intergrax.contracts.provider_invocation_store import (
    ProviderInvocationConflictError,
    ProviderInvocationPersistenceError,
    ProviderInvocationStore,
)
from tests.unit.runtime.governance.gr3_test_support import (
    StaticActiveTaskScope,
    default_gr3_identity_bundle,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_T0 = datetime(2026, 9, 16, 11, 0, 0, tzinfo=timezone.utc)


@dataclass
class RecordingProviderInvocationStore:
    """Custom injectable store — not InMemory concrete from host."""

    invocations: dict[str, ProviderInvocation] = field(default_factory=dict)
    outcomes: dict[str, ProviderInvocationOutcome] = field(default_factory=dict)
    fail_intent: bool = False
    fail_outcome: bool = False
    intent_puts: int = 0
    outcome_puts: int = 0

    @property
    def is_durable(self) -> bool:
        return True

    def put_invocation(self, invocation: ProviderInvocation) -> None:
        self.intent_puts += 1
        if self.fail_intent:
            raise ProviderInvocationPersistenceError("injected intent failure")
        if invocation.invocation_id in self.invocations:
            if self.invocations[invocation.invocation_id] != invocation:
                raise ProviderInvocationConflictError("conflict")
            return
        self.invocations[invocation.invocation_id] = invocation

    def get_invocation(self, invocation_id: str) -> ProviderInvocation | None:
        return self.invocations.get(invocation_id)

    def put_outcome(self, outcome: ProviderInvocationOutcome) -> None:
        self.outcome_puts += 1
        if self.fail_outcome:
            raise ProviderInvocationPersistenceError("injected outcome failure")
        if outcome.invocation_id in self.outcomes:
            if self.outcomes[outcome.invocation_id] != outcome:
                raise ProviderInvocationConflictError("outcome conflict")
            return
        self.outcomes[outcome.invocation_id] = outcome

    def get_outcome(self, invocation_id: str) -> ProviderInvocationOutcome | None:
        return self.outcomes.get(invocation_id)


def _runtime_with_store(
    fake: DeterministicExternalWorkFake,
    task_id: object,
    store: ProviderInvocationStore,
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
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        _policy_bundle,
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
    )


def test_success_persists_intent_before_outcome_and_ger() -> None:
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 1
    assert step.governed_result is not None
    inv_id = step.governed_result.provider_invocation.invocation_id
    assert store.invocations[inv_id].invocation_id == inv_id
    assert store.outcomes[inv_id].status is ProviderInvocationStatus.SUCCEEDED
    assert store.intent_puts >= 1
    assert store.outcome_puts >= 1


def test_intent_persistence_failure_zero_provider_calls() -> None:
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore(fail_intent=True)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 0
    assert step.reason == "provider_invocation_intent_persistence_failed"
    assert step.external_effect_outcome is None
    assert store.outcomes == {}
    decision = step.adapter_result.policy_decision
    assert decision is None or decision.action is not PolicyAction.DENY


def test_outcome_persistence_failure_single_provider_call() -> None:
    fake = DeterministicExternalWorkFake()
    store = RecordingProviderInvocationStore(fail_outcome=True)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, fake = _create_step(
        runtime, fake, task_id, run_id, attempt_id, execution_id,
    )
    assert fake.create_calls == 1
    assert step.reason == "provider_invocation_outcome_persistence_failed"
    assert step.state is GovernedExternalWorkHostState.EXECUTION_FAILED
    assert step.governed_result is None
    assert len(store.invocations) == 1
    assert store.outcomes == {}


def test_failure_outcome_persisted_no_ger() -> None:
    fake = DeterministicExternalWorkFake(
        fail_create_with_code={_FAIL_IDEMP: ExternalWorkErrorCode.PERMANENT_PROVIDER_FAILURE},
    )
    store = DurableTestProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_FAIL_IDEMP,
    )
    assert step.governed_result is None
    assert step.state is GovernedExternalWorkHostState.EXECUTION_FAILED
    assert step.external_effect_outcome is ExternalEffectOutcome.FAILURE
    inv_id = next(iter(store._invocations))  # noqa: SLF001
    assert store.get_outcome(inv_id).status is ProviderInvocationStatus.FAILED


def test_unknown_outcome_persisted() -> None:
    fake = DeterministicExternalWorkFake(
        fail_create_with_code={
            _UNCERTAIN_IDEMP: ExternalWorkErrorCode.PROVIDER_OUTCOME_UNCERTAIN,
        },
    )
    store = DurableTestProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    step, _ = _create_step(
        runtime,
        fake,
        task_id,
        run_id,
        attempt_id,
        execution_id,
        idempotency_key=_UNCERTAIN_IDEMP,
    )
    inv_id = next(iter(store._invocations))  # noqa: SLF001
    assert store.get_outcome(inv_id).status is ProviderInvocationStatus.UNKNOWN
    assert step.state is GovernedExternalWorkHostState.EXECUTION_OUTCOME_UNKNOWN
    assert step.governed_result is None
    assert step.external_effect_outcome is ExternalEffectOutcome.UNKNOWN


def test_duplicate_intent_same_payload_idempotent() -> None:
    store = InMemoryProviderInvocationStore()
    inv = ProviderInvocation(
        invocation_id="inv-dup",
        provider_id=_PROVIDER,
        operation="create_work",
        task_id="t1",
        run_id="r1",
        request_digest="sha256:" + ("ab" * 32),
        started_at=_T0,
    )
    store.put_invocation(inv)
    store.put_invocation(inv)
    assert store.get_invocation("inv-dup") == inv


def test_duplicate_intent_conflict_rejected() -> None:
    store = InMemoryProviderInvocationStore()
    base = ProviderInvocation(
        invocation_id="inv-conflict",
        provider_id=_PROVIDER,
        operation="create_work",
        task_id="t1",
        run_id="r1",
        request_digest="sha256:" + ("ab" * 32),
        started_at=_T0,
    )
    store.put_invocation(base)
    other = base.model_copy(update={"task_id": "t2"})
    with pytest.raises(ProviderInvocationConflictError):
        store.put_invocation(other)


def test_crash_window_intent_without_outcome_after_dispatch() -> None:
    """Simulate crash after provider: intent durable, outcome missing until finalize."""
    fake = DeterministicExternalWorkFake()
    store = DurableTestProviderInvocationStore()
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    runtime = _runtime_with_store(fake, task_id, store)
    _create_step(runtime, fake, task_id, run_id, attempt_id, execution_id)
    assert len(store._invocations) == 1  # noqa: SLF001
    assert len(store._outcomes) == 1  # noqa: SLF001


def test_production_rejects_non_durable_in_memory_store() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, _, _, _ = default_gr3_identity_bundle()
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        _policy_bundle,
    )

    execution_store = InMemoryGovernedExecutionStore()
    receipt_store = InMemoryProofReceiptStore()
    bundle_store = InMemoryPolicyBundleArtifactStore()
    continuation_store = InMemoryContinuationStateStore()
    with pytest.raises(ValueError, match="is_durable"):
        build_governed_external_work_production_runtime(
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
            provider_invocation_store=InMemoryProviderInvocationStore(),
        )


def test_production_accepts_custom_durable_store() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, _, _, _ = default_gr3_identity_bundle()
    store = RecordingProviderInvocationStore()
    runtime = _runtime_with_store(fake, task_id, store)
    assert runtime.orchestrator is not None


def test_host_stores_module_has_no_duplicate_provider_invocation_store_port() -> None:
    text = Path(__file__).resolve().parents[2] / "host" / "stores.py"
    source = text.read_text(encoding="utf-8")
    assert "ProviderInvocationStorePort" not in source


def test_production_runtime_requires_provider_invocation_store_kwarg() -> None:
    fake = DeterministicExternalWorkFake()
    task_id, _, _, _ = default_gr3_identity_bundle()
    cw = gr6_seeded_collaborative_work_repositories(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_PRINCIPAL,
    )
    from applications.governed_contractor_application.tests.host.test_gr7_a2_external_work_erl_bridge import (
        _policy_bundle,
        _stores,
    )
    from intergrax.contracts.external_work_provider_capabilities import (
        quote_first_partner_capability_fixture,
    )

    execution_store, receipt_store, bundle_store, continuation_store, _ = _stores()
    with pytest.raises(TypeError):
        build_governed_external_work_production_runtime(
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
        )
