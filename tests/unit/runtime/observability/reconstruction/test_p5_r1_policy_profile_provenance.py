# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P5-R1 policy and profile provenance reconstruction integrity."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from intergrax.applications._shared.profile_resolution.execution_pinning import (
    InMemoryEffectiveProfileExecutionPinningStore,
)
from intergrax.applications._shared.profile_resolution.execution_effective_profile_provenance_reader import (
    PinningStoreExecutionEffectiveProfileProvenanceReader,
)
from intergrax.applications.contracts.profile_resolution.execution_binding import (
    EffectiveProfileExecutionBinding,
)
from intergrax.applications.contracts.profile_resolution.revision_id import (
    mint_effective_profile_revision_id,
)
from intergrax.contracts.execution_effective_profile_provenance import (
    ExecutionEffectiveProfileProvenanceReadStatus,
)
from intergrax.contracts.execution_event_position import (
    ExecutionEventPosition,
)
from intergrax.contracts.positioned_runtime_event import (
    PositionedRuntimeEvent,
    as_of_boundary_for_positioned,
)
from intergrax.runtime.observability.reconstruction.policy_provenance_projection import (
    project_policy_decision_provenance,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    require_active_execution_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.events.runtime_event import RuntimeEvent
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import (
    ExecutionReconstructionIntegrityError,
    ExecutionReconstructor,
)

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"


def _pin(
    pinning: InMemoryEffectiveProfileExecutionPinningStore,
    *,
    tenant_id: str,
    execution_id: ExecutionId,
    revision_id_value: str | None = None,
    fingerprint: str = "fp-test-1",
) -> None:
    revision_id = revision_id_value or mint_effective_profile_revision_id()
    pinning.pin(
        EffectiveProfileExecutionBinding(
            tenant_id=tenant_id,
            execution_id=execution_id,
            revision_id=revision_id,
            fingerprint=fingerprint,
        )
    )


def _policy_event(
    *,
    tenant_id: str,
    task_id: TaskId,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
    bundle_id: str = "bundle-1",
    position: int = 1,
) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=mint_event_id(),
        tenant_id=tenant_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        event_type=RuntimeEventType.POLICY_DECISION,
        phase=ExecutionPhase.STEP_EXECUTION,
        timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        payload={
            "governance_evidence_schema": "governance_decision_evidence.v1",
            "evidence_id": "ev-1",
            "evaluation_point": "PRE_MODEL",
            "action": "allow",
            "resource_type": "execution",
            "resource_scope": "task",
            "decision": "allow",
            "reason": "ok",
            "reason_code": "OK",
            "policy_bundle_id": bundle_id,
            "policy_bundle_version": "1.0.0",
            "policy_bundle_digest": "digest-abc",
            "policy_rule_id": "rule-42",
            "request_digest": "req-digest",
            "idempotency_key": "idem-1",
            "workspace_id": "ws",
            "principal_id": "principal",
            "_position": position,
        },
    )


def test_policy_provenance_fields_and_order() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            bundle_id="bundle-first",
            position=1,
        ),
        tenant_id=_TENANT_A,
    )
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            bundle_id="bundle-second",
            position=2,
        ),
        tenant_id=_TENANT_A,
    )
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert len(view.policy_decision_provenance) == 2
    assert view.policy_decision_provenance[0].policy_bundle_id == "bundle-first"
    assert view.policy_decision_provenance[1].policy_bundle_id == "bundle-second"
    assert view.policy_decision_provenance[0].policy_rule_id == "rule-42"
    assert view.policy_decision_provenance[0].evaluation_point == "PRE_MODEL"
    assert view.policy_decision_provenance[0].execution_id == execution_id


def test_policy_cross_tenant_rejected() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    event = _policy_event(
        tenant_id=_TENANT_B,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    positioned = (
        PositionedRuntimeEvent(event=event, position=ExecutionEventPosition(1)),
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        project_policy_decision_provenance(
            positioned,
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
        )


def test_policy_malformed_payload_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            event_type=RuntimeEventType.POLICY_DECISION,
            phase=ExecutionPhase.STEP_EXECUTION,
            timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            payload={"evidence_id": ""},
        ),
        tenant_id=_TENANT_A,
    )
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)


def test_profile_reader_absent_does_not_invent_provenance() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert view.execution_effective_profile_provenance == ()
    assert (
        view.effective_profile_provenance_read_status
        is ExecutionEffectiveProfileProvenanceReadStatus.NOT_CONFIGURED
    )


def test_profile_missing_binding_fails_when_reader_configured() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
        tenant_id=_TENANT_A,
    )
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(
        InMemoryEffectiveProfileExecutionPinningStore(),
    )
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)


def test_profile_cross_tenant_binding_invisible() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    _pin(pinning, tenant_id=_TENANT_A, execution_id=execution_id)
    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_B,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
        ),
        tenant_id=_TENANT_B,
    )
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_B, task_id, run_id)


def test_profile_as_of_excludes_future_execution_id() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()
    execution_a = mint_execution_id()
    execution_b = mint_execution_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    for execution_id in (execution_a, execution_b):
        _pin(pinning, tenant_id=_TENANT_A, execution_id=execution_id)
    store = InMemoryRuntimeEventStore()
    event_a = _policy_event(
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_a,
        execution_id=execution_a,
        position=1,
    )
    event_b = _policy_event(
        tenant_id=_TENANT_A,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_b,
        execution_id=execution_b,
        position=2,
    )
    positioned_a = store.append(event_a, tenant_id=_TENANT_A)
    store.append(event_b, tenant_id=_TENANT_A)
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    boundary = as_of_boundary_for_positioned(positioned_a)
    view = reconstructor.reconstruct_execution(
        _TENANT_A,
        task_id,
        run_id,
        execution_as_of=boundary,
    )
    assert len(view.execution_effective_profile_provenance) == 1
    assert view.execution_effective_profile_provenance[0].execution_id == execution_a


def test_reconstruction_does_not_invoke_governance_evaluator() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    store = InMemoryRuntimeEventStore()
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    evaluator = MagicMock()
    reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    evaluator.assert_not_called()


def test_policy_provenance_from_canonical_governance_persistence_chain() -> None:
    from intergrax.contracts.governed_execution_governance_evidence import (
        GovernedExecutionEvaluationPoint,
        GovernanceDecisionEvidenceFact,
    )
    from intergrax.contracts.runtime_policy import PolicyAction
    from intergrax.runtime.events.evidence_persistence_adapter import (
        as_evidence_persistence_port,
    )
    from intergrax.runtime.events.payloads.spine_families import PolicyDecisionSpinePayloadV1
    from intergrax.runtime.governance.governance_evidence_persistence import (
        RuntimeEventGovernanceEvidencePersistence,
    )

    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    evidence_port = as_evidence_persistence_port(store)
    assert evidence_port is not None
    governance_persistence = RuntimeEventGovernanceEvidencePersistence(
        evidence_persistence=evidence_port,
    )
    fact = GovernanceDecisionEvidenceFact(
        evidence_id="ev-canonical-1",
        recorded_at=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        tenant_id=_TENANT_A,
        workspace_id="ws-1",
        principal_id="principal-1",
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        evaluation_point=GovernedExecutionEvaluationPoint.PRE_MODEL,
        action="allow",
        decision=PolicyAction.ALLOW,
        policy_bundle_id="bundle-canonical",
        policy_bundle_version="2.0.0",
        policy_bundle_digest="sha256:" + ("a" * 64),
        policy_rule_id="rule-canonical",
        request_digest="sha256:" + ("b" * 64),
        idempotency_key="idem-canonical",
    )
    outcome = governance_persistence.persist(fact)
    assert outcome.persisted
    persisted = store.list_for_task(str(task_id), tenant_id=_TENANT_A)
    policy_events = [
        event
        for event in persisted
        if event.event_type is RuntimeEventType.POLICY_DECISION
    ]
    assert len(policy_events) == 1
    assert (
        policy_events[0].payload.get("payload_schema_id")
        == PolicyDecisionSpinePayloadV1.schema_id
    )
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    view = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert len(view.policy_decision_provenance) == 1
    provenance = view.policy_decision_provenance[0]
    assert provenance.evidence_id == "ev-canonical-1"
    assert provenance.evaluation_point == "pre_model"
    assert provenance.action == "allow"
    assert provenance.decision == "allow"
    assert provenance.policy_bundle_id == "bundle-canonical"
    assert provenance.policy_bundle_version == "2.0.0"
    assert provenance.policy_bundle_digest == "sha256:" + ("a" * 64)
    assert provenance.policy_rule_id == "rule-canonical"
    assert provenance.execution_id == execution_id
    assert provenance.tenant_id == _TENANT_A
    assert provenance.task_id == task_id
    assert provenance.run_id == run_id
    assert provenance.attempt_id == attempt_id


def test_policy_typed_envelope_malformed_fails_closed() -> None:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    store = InMemoryRuntimeEventStore()
    store.append(
        RuntimeEvent(
            event_id=mint_event_id(),
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=execution_id,
            event_type=RuntimeEventType.POLICY_DECISION,
            phase=ExecutionPhase.STEP_EXECUTION,
            timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
            payload={
                "payload_schema_id": "policy_decision_spine.v1",
                "data": {"evidence_id": ""},
            },
        ),
        tenant_id=_TENANT_A,
    )
    reconstructor = ExecutionReconstructor(store, InMemoryCausalEvidencePersistence())
    with pytest.raises(ExecutionReconstructionIntegrityError):
        reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)


@pytest.mark.asyncio
async def test_child_profile_inheritance_pins_child_to_parent_revision() -> None:
    from intergrax.applications._shared.profile_resolution.profile_resolution_child_context_inheritance_adapter import (
        ProfileResolutionChildContextInheritanceAdapter,
    )
    from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority
    from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
    from intergrax.runtime.execution.active_execution_budget import (
        bind_root_execution_budget,
        reset_active_execution_budget,
    )
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
    from intergrax.runtime.execution.child import ChildExecutionRunner
    from intergrax.runtime.governance.active_execution_authority import (
        bind_active_execution_authority,
        reset_active_execution_authority,
    )
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        reset_active_execution_identity,
    )
    from intergrax.runtime.nexus.budget.budget_models import RunBudget

    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution_id = mint_execution_id()
    revision_p1 = mint_effective_profile_revision_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    _pin(
        pinning,
        tenant_id=_TENANT_A,
        execution_id=root_execution_id,
        revision_id_value=revision_p1,
    )
    adapter = ProfileResolutionChildContextInheritanceAdapter(
        tenant_id=_TENANT_A,
        pinning_store=pinning,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[object, str](
        ledger=ledger,
        child_context_inheritance=adapter,
    )
    child_execution_ids: list[ExecutionId] = []

    class _ChildDelegate:
        async def execute(self, request: object) -> str:
            child_execution_ids.append(require_active_execution_id())
            return "ok"

    class _RootDelegate:
        async def execute(self, request: object) -> str:
            return await child_runner.execute(request=request, delegate=_ChildDelegate())

    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=root_execution_id,
        task_id=task_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=root_execution_id,
        ledger=ledger,
    )
    try:
        await ExecutionBoundary(
            _RootDelegate(),
            identity=ExecutionIdentityBinding(
                run_id=run_id,
                attempt_id=attempt_id,
                execution_id=root_execution_id,
                task_id=task_id,
            ),
            authority=authority,
        ).execute(None)
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)
        reset_active_execution_authority(authority_token)

    assert len(child_execution_ids) == 1
    child_execution_id = child_execution_ids[0]
    root_binding = pinning.get(tenant_id=_TENANT_A, execution_id=root_execution_id)
    child_binding = pinning.get(tenant_id=_TENANT_A, execution_id=child_execution_id)
    assert root_binding is not None
    assert child_binding is not None
    assert root_binding.revision_id == revision_p1
    assert child_binding.revision_id == revision_p1

    store = InMemoryRuntimeEventStore()
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=root_execution_id,
            position=1,
        ),
        tenant_id=_TENANT_A,
    )
    store.append(
        _policy_event(
            tenant_id=_TENANT_A,
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            execution_id=child_execution_id,
            position=2,
        ),
        tenant_id=_TENANT_A,
    )
    reader = PinningStoreExecutionEffectiveProfileProvenanceReader(pinning)
    reconstructor = ExecutionReconstructor(
        store,
        InMemoryCausalEvidencePersistence(),
        execution_effective_profile_provenance_reader=reader,
    )
    view = reconstructor.reconstruct_execution(_TENANT_A, task_id, run_id)
    assert len(view.execution_effective_profile_provenance) == 2
    revisions = {
        item.execution_id: item.revision_ref.value
        for item in view.execution_effective_profile_provenance
    }
    assert revisions[root_execution_id] == revision_p1.value
    assert revisions[child_execution_id] == revision_p1.value


@pytest.mark.asyncio
async def test_child_profile_inheritance_missing_parent_binding_fails_before_delegate() -> None:
    from intergrax.applications._shared.profile_resolution.profile_resolution_child_context_inheritance_adapter import (
        ProfileResolutionChildContextInheritanceAdapter,
    )
    from intergrax.applications.contracts.profile_resolution.errors import (
        MissingPinnedEffectiveProfileRevisionError,
    )
    from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority
    from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
    from intergrax.runtime.execution.child import ChildExecutionRunner
    from intergrax.runtime.execution.active_execution_budget import (
        bind_root_execution_budget,
        reset_active_execution_budget,
    )
    from intergrax.runtime.governance.active_execution_authority import (
        bind_active_execution_authority,
        reset_active_execution_authority,
    )
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        reset_active_execution_identity,
    )
    from intergrax.runtime.nexus.budget.budget_models import RunBudget

    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution_id = mint_execution_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    adapter = ProfileResolutionChildContextInheritanceAdapter(
        tenant_id=_TENANT_A,
        pinning_store=pinning,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[object, str](
        ledger=ledger,
        child_context_inheritance=adapter,
    )
    delegate_calls = 0

    class _ChildDelegate:
        async def execute(self, request: object) -> str:
            nonlocal delegate_calls
            delegate_calls += 1
            return "ok"

    class _RootDelegate:
        async def execute(self, request: object) -> str:
            return await child_runner.execute(request=request, delegate=_ChildDelegate())

    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=root_execution_id,
        task_id=task_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=root_execution_id,
        ledger=ledger,
    )
    try:
        with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
            await ExecutionBoundary(
                _RootDelegate(),
                identity=ExecutionIdentityBinding(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=root_execution_id,
                    task_id=task_id,
                ),
                authority=authority,
            ).execute(None)
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)
        reset_active_execution_authority(authority_token)
    assert delegate_calls == 0


@pytest.mark.asyncio
async def test_child_profile_inheritance_cross_tenant_parent_invisible() -> None:
    from intergrax.applications._shared.profile_resolution.profile_resolution_child_context_inheritance_adapter import (
        ProfileResolutionChildContextInheritanceAdapter,
    )
    from intergrax.applications.contracts.profile_resolution.errors import (
        MissingPinnedEffectiveProfileRevisionError,
    )
    from intergrax.contracts.delegation_authority import resolve_root_parent_execution_authority
    from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
    from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
    from intergrax.runtime.execution.child import ChildExecutionRunner
    from intergrax.runtime.execution.active_execution_budget import (
        bind_root_execution_budget,
        reset_active_execution_budget,
    )
    from intergrax.runtime.governance.active_execution_authority import (
        bind_active_execution_authority,
        reset_active_execution_authority,
    )
    from intergrax.contracts.execution_identity import (
        bind_active_execution_identity,
        reset_active_execution_identity,
    )
    from intergrax.runtime.nexus.budget.budget_models import RunBudget

    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution_id = mint_execution_id()
    pinning = InMemoryEffectiveProfileExecutionPinningStore()
    _pin(pinning, tenant_id=_TENANT_A, execution_id=root_execution_id)
    adapter = ProfileResolutionChildContextInheritanceAdapter(
        tenant_id=_TENANT_B,
        pinning_store=pinning,
    )
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[object, str](
        ledger=ledger,
        child_context_inheritance=adapter,
    )

    class _ChildDelegate:
        async def execute(self, request: object) -> str:
            return "ok"

    class _RootDelegate:
        async def execute(self, request: object) -> str:
            return await child_runner.execute(request=request, delegate=_ChildDelegate())

    authority = resolve_root_parent_execution_authority(None)
    authority_token = bind_active_execution_authority(authority)
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=root_execution_id,
        task_id=task_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=root_execution_id,
        ledger=ledger,
    )
    try:
        with pytest.raises(MissingPinnedEffectiveProfileRevisionError):
            await ExecutionBoundary(
                _RootDelegate(),
                identity=ExecutionIdentityBinding(
                    run_id=run_id,
                    attempt_id=attempt_id,
                    execution_id=root_execution_id,
                    task_id=task_id,
                ),
                authority=authority,
            ).execute(None)
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)
        reset_active_execution_authority(authority_token)
    assert pinning.get(tenant_id=_TENANT_B, execution_id=root_execution_id) is None
