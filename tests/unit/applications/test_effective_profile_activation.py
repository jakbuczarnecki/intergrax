# © Artur Czarnecki. All rights reserved.

"""P1.6 — atomic effective profile activation lifecycle."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from intergrax.applications._shared.profile_resolution import (
    EffectiveProfileActivationDependencies,
    EffectiveProfileActivationService,
    EffectiveProfileExecutionPinningDependencies,
    InMemoryActiveEffectiveProfileRevisionStore,
    InMemoryEffectiveProfileExecutionPinningStore,
    InMemoryEffectiveProfileRevisionStore,
    activate_materialized_revision,
    attach_revision_checkpoint_evidence_to_task,
    build_effective_profile_revision_admission,
    inherit_child_execution_pinned_revision,
    materialize_effective_profile_revision,
    pin_effective_profile_revision_for_execution,
    require_execution_pinned_revision,
    resolve_active_effective_profile_revision,
    resolve_profile,
    resolve_revision_for_execution,
)
from intergrax.applications._shared.profile_resolution.activation_store import (
    KvActiveEffectiveProfileRevisionStore,
)
from intergrax.applications._shared.runtime_inspection.service import RuntimeInspectionService
from intergrax.applications.contracts.capability_health import CapabilityHealthStatus
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.profile_resolution import (
    ActivateEffectiveProfileRevisionRequest,
    EffectiveProfileActivationConflictError,
    EffectiveProfileActivationPersistenceError,
    EffectiveProfileActivationRejectedError,
    EffectiveProfileActivationRevisionNotFoundError,
    EffectiveProfileActivationScopeMismatchError,
    EffectiveProfileRevisionScope,
    mint_effective_profile_revision_id,
)
from intergrax.applications.contracts.runtime_inspection.completeness import InspectionCompleteness
from intergrax.contracts.execution_identity import ExecutionId, mint_execution_id, mint_task_id
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.runtime.task.task import Task, TaskContext
from tests.unit.runtime.background_execution.reentry_admission_doubles import InMemoryKVStore

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

_SCOPE_A = EffectiveProfileRevisionScope(application_id="activation.test", tenant_id="tenant-a")
_SCOPE_B = EffectiveProfileRevisionScope(application_id="activation.test", tenant_id="tenant-b")


def _application(**updates: object) -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="activation.test")
    if not updates:
        return profile
    return profile.model_copy(update=updates)


def _revision(
    application: ApplicationEnvironmentProfile | None = None,
    *,
    scope: EffectiveProfileRevisionScope = _SCOPE_A,
    store: InMemoryEffectiveProfileRevisionStore | None = None,
    predecessor: object | None = None,
) -> tuple[object, object]:
    revision_store = store or InMemoryEffectiveProfileRevisionStore()
    resolution = resolve_profile(application or _application(), layers=())
    revision = materialize_effective_profile_revision(
        resolution,
        scope=scope,
        store=revision_store,
        predecessor_revision_id=(
            predecessor.revision_id if predecessor is not None else None
        ),
    )
    return revision_store, revision


def _service(
    revision_store: InMemoryEffectiveProfileRevisionStore,
    active_store: InMemoryActiveEffectiveProfileRevisionStore | None = None,
    *,
    eligibility_checker: object | None = None,
) -> EffectiveProfileActivationService:
    return EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=revision_store,
            active_store=active_store or InMemoryActiveEffectiveProfileRevisionStore(),
            eligibility_checker=eligibility_checker,
        ),
    )


def _activate(
    service: EffectiveProfileActivationService,
    revision: object,
    *,
    scope: EffectiveProfileRevisionScope = _SCOPE_A,
    expected: object | None = None,
) -> object:
    expected_id = expected.revision_id if expected is not None else None
    if expected is ...:
        current = service.get_active_binding(scope)
        expected_id = current.revision_id if current is not None else None
    return service.activate(
        ActivateEffectiveProfileRevisionRequest(
            scope=scope,
            candidate_revision_id=revision.revision_id,
            expected_active_revision_id=expected_id,
        ),
    )


def _admission_dependencies(
    revision_store: InMemoryEffectiveProfileRevisionStore,
    pinning_store: InMemoryEffectiveProfileExecutionPinningStore,
    active_store: InMemoryActiveEffectiveProfileRevisionStore,
    *,
    scope: EffectiveProfileRevisionScope = _SCOPE_A,
) -> EffectiveProfileExecutionPinningDependencies:
    return EffectiveProfileExecutionPinningDependencies(
        revision_store=revision_store,
        pinning_store=pinning_store,
        active_store=active_store,
        scope=scope,
    )


def _echo_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-a",
        user_id="user-1",
        message="activation proof",
        context=TaskContext(capability="echo.basic"),
        agent_id="echo",
    )


def test_first_activation_none_to_r1() -> None:
    revision_store, revision = _revision()
    service = _service(revision_store)
    result = _activate(service, revision, expected=None)
    active = service.get_active_binding(_SCOPE_A)
    assert result.changed is True
    assert result.previous_revision_id is None
    assert result.active_revision_id == revision.revision_id
    assert active is not None
    assert active.revision_id == revision.revision_id
    assert active.fingerprint == revision.fingerprint


def test_normal_activation_r1_to_r2() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    service = _service(revision_store)
    _activate(service, revision_r1, expected=None)
    result = _activate(service, revision_r2, expected=revision_r1)
    assert result.changed is True
    assert result.previous_revision_id == revision_r1.revision_id
    assert service.get_active_binding(_SCOPE_A).revision_id == revision_r2.revision_id


def test_stale_cas_conflict() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    _, revision_r3 = _revision(store=revision_store, predecessor=revision_r2)
    service = _service(revision_store)
    _activate(service, revision_r1, expected=None)
    _activate(service, revision_r2, expected=revision_r1)
    with pytest.raises(EffectiveProfileActivationConflictError):
        _activate(service, revision_r3, expected=revision_r1)
    assert service.get_active_binding(_SCOPE_A).revision_id == revision_r2.revision_id


def test_candidate_missing_fails_closed() -> None:
    revision_store, _existing = _revision()
    service = _service(revision_store)
    missing_id = mint_effective_profile_revision_id()
    with pytest.raises(EffectiveProfileActivationRevisionNotFoundError):
        service.activate(
            ActivateEffectiveProfileRevisionRequest(
                scope=_SCOPE_A,
                candidate_revision_id=missing_id,
                expected_active_revision_id=None,
            ),
        )
    assert service.get_active_binding(_SCOPE_A) is None


def test_scope_mismatch_fails_closed() -> None:
    revision_store, revision = _revision(scope=_SCOPE_A)
    service = _service(revision_store)
    with pytest.raises(EffectiveProfileActivationRevisionNotFoundError):
        _activate(service, revision, scope=_SCOPE_B, expected=None)


def test_activation_does_not_mutate_revisions() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    before_r1 = revision_r1.model_dump_json()
    before_r2 = revision_r2.model_dump_json()
    service = _service(revision_store)
    _activate(service, revision_r1, expected=None)
    _activate(service, revision_r2, expected=revision_r1)
    assert revision_r1.model_dump_json() == before_r1
    assert revision_r2.model_dump_json() == before_r2


def test_idempotent_same_revision_changed_false() -> None:
    revision_store, revision = _revision()
    service = _service(revision_store)
    first = _activate(service, revision, expected=None)
    second = _activate(service, revision, expected=revision)
    assert first.changed is True
    assert second.changed is False


def test_rollback_r2_to_r1() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    service = _service(revision_store)
    _activate(service, revision_r1, expected=None)
    _activate(service, revision_r2, expected=revision_r1)
    result = service.rollback(
        scope=_SCOPE_A,
        target_revision_id=revision_r1.revision_id,
        expected_active_revision_id=revision_r2.revision_id,
    )
    assert result.active_revision_id == revision_r1.revision_id
    assert service.get_active_binding(_SCOPE_A).revision_id == revision_r1.revision_id


def test_rollback_conflict() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    _, revision_r3 = _revision(store=revision_store, predecessor=revision_r2)
    service = _service(revision_store)
    _activate(service, revision_r1, expected=None)
    _activate(service, revision_r2, expected=revision_r1)
    _activate(service, revision_r3, expected=revision_r2)
    with pytest.raises(EffectiveProfileActivationConflictError):
        service.rollback(
            scope=_SCOPE_A,
            target_revision_id=revision_r1.revision_id,
            expected_active_revision_id=revision_r2.revision_id,
        )


def test_execution_pinning_r1_stays_after_r2_activation() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision_r1, expected=None)
    execution_id = mint_execution_id()
    admission = build_effective_profile_revision_admission(
        _admission_dependencies(revision_store, pinning_store, active_store),
    )
    admission.admit_root_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        task=_echo_task(),
    )
    _activate(service, revision_r2, expected=revision_r1)
    binding = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
    )
    assert binding.revision_id == revision_r1.revision_id
    execution_id_2 = mint_execution_id()
    admission.admit_root_execution(
        tenant_id="tenant-a",
        execution_id=execution_id_2,
        task=_echo_task(),
    )
    binding_2 = require_execution_pinned_revision(
        tenant_id="tenant-a",
        execution_id=execution_id_2,
        pinning_store=pinning_store,
    )
    assert binding_2.revision_id == revision_r2.revision_id


def test_child_inherits_parent_pinned_revision_after_activation() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision_r1, expected=None)
    parent_execution_id = mint_execution_id()
    pin_effective_profile_revision_for_execution(
        revision=revision_r1,
        tenant_id="tenant-a",
        execution_id=parent_execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
    )
    _activate(service, revision_r2, expected=revision_r1)
    child_execution_id = mint_execution_id()
    child_binding = inherit_child_execution_pinned_revision(
        tenant_id="tenant-a",
        parent_execution_id=parent_execution_id,
        child_execution_id=child_execution_id,
        pinning_store=pinning_store,
    )
    assert child_binding.revision_id == revision_r1.revision_id


def test_resume_uses_pinned_revision_not_active() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision_r1, expected=None)
    execution_id = mint_execution_id()
    admission = build_effective_profile_revision_admission(
        _admission_dependencies(revision_store, pinning_store, active_store),
    )
    admitted_task = admission.admit_root_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        task=_echo_task(),
    )
    _activate(service, revision_r2, expected=revision_r1)
    checkpoint = TaskCheckpoint(
        task_id=admitted_task.task_id,
        tenant_id="tenant-a",
        resume_token="resume-proof",
        task_state=admitted_task.state,
        task_snapshot=admitted_task.model_dump(mode="json"),
    )
    resumed = admission.admit_root_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        task=_echo_task(),
        resume_checkpoint=checkpoint,
    )
    del resumed
    resolved = resolve_revision_for_execution(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning_store,
        revision_store=revision_store,
        scope_application_id=_SCOPE_A.application_id,
        scope_tenant_id=_SCOPE_A.tenant_id,
    )
    assert resolved.revision_id == revision_r1.revision_id


def test_admission_activation_race_produces_coherent_pinning() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    pinning_store = InMemoryEffectiveProfileExecutionPinningStore()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision_r1, expected=None)
    admission = build_effective_profile_revision_admission(
        _admission_dependencies(revision_store, pinning_store, active_store),
    )
    barrier = threading.Barrier(2)
    errors: list[Exception] = []

    def _admit() -> None:
        try:
            barrier.wait(timeout=5)
            execution_id = mint_execution_id()
            admission.admit_root_execution(
                tenant_id="tenant-a",
                execution_id=execution_id,
                task=_echo_task(),
            )
            binding = require_execution_pinned_revision(
                tenant_id="tenant-a",
                execution_id=execution_id,
                pinning_store=pinning_store,
            )
            revision = resolve_revision_for_execution(
                tenant_id="tenant-a",
                execution_id=execution_id,
                pinning_store=pinning_store,
                revision_store=revision_store,
                scope_application_id=_SCOPE_A.application_id,
                scope_tenant_id=_SCOPE_A.tenant_id,
            )
            assert binding.revision_id == revision.revision_id
            assert binding.fingerprint == revision.fingerprint
        except Exception as exc:
            errors.append(exc)

    def _activate_r2() -> None:
        try:
            barrier.wait(timeout=5)
            _activate(service, revision_r2, expected=revision_r1)
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_admit), pool.submit(_activate_r2)]
        for future in as_completed(futures):
            future.result()
    assert not errors


def test_concurrent_cas_exactly_one_winner() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    _, revision_r3 = _revision(store=revision_store, predecessor=revision_r1)
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision_r1, expected=None)
    barrier = threading.Barrier(2)
    results: list[object] = []
    errors: list[Exception] = []

    def _try_activate(revision: object) -> None:
        try:
            barrier.wait(timeout=5)
            results.append(
                _activate(service, revision, expected=revision_r1),
            )
        except EffectiveProfileActivationConflictError as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(_try_activate, revision_r2),
            pool.submit(_try_activate, revision_r3),
        ]
        for future in as_completed(futures):
            future.result()
    assert len(results) == 1
    assert len(errors) == 1
    active = service.get_active_binding(_SCOPE_A)
    assert active.revision_id in {revision_r2.revision_id, revision_r3.revision_id}


def test_persistence_failure_leaves_prior_active() -> None:
    revision_store, revision_r1 = _revision()
    _, revision_r2 = _revision(store=revision_store, predecessor=revision_r1)
    active_store = InMemoryActiveEffectiveProfileRevisionStore(
        persist_hook=lambda: (_ for _ in ()).throw(RuntimeError("persist failed")),
    )
    service = _service(revision_store, active_store)
    _activate(service, revision_r1, expected=None)
    with pytest.raises(EffectiveProfileActivationPersistenceError):
        _activate(service, revision_r2, expected=revision_r1)
    assert service.get_active_binding(_SCOPE_A).revision_id == revision_r1.revision_id


def test_p13_validation_reuse_blocks_invalid_candidate() -> None:
    revision_store, revision = _revision()
    service = _service(
        revision_store,
        eligibility_checker=lambda _candidate: (_ for _ in ()).throw(
            EffectiveProfileActivationRejectedError("required dependency unavailable"),
        ),
    )
    with pytest.raises(EffectiveProfileActivationRejectedError):
        _activate(service, revision, expected=None)
    assert service.get_active_binding(_SCOPE_A) is None


def test_p15_health_reuse_blocks_unavailable_candidate() -> None:
    revision_store, revision = _revision()
    service = _service(
        revision_store,
        eligibility_checker=lambda _candidate: (_ for _ in ()).throw(
            EffectiveProfileActivationRejectedError(
                f"health={CapabilityHealthStatus.UNAVAILABLE.value}",
            ),
        ),
    )
    with pytest.raises(EffectiveProfileActivationRejectedError, match="unavailable"):
        _activate(service, revision, expected=None)


def test_inspection_exposes_active_revision() -> None:
    revision_store, revision = _revision()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision, expected=None)
    inspection = RuntimeInspectionService(
        revision_store=revision_store,
        active_store=active_store,
    ).inspect_active_revision(scope=_SCOPE_A)
    assert inspection.completeness is InspectionCompleteness.COMPLETE
    assert inspection.revision is not None
    assert inspection.revision.revision_id == revision.revision_id
    assert inspection.safe_revision is not None
    payload = json.loads(inspection.model_dump_json())
    assert "sk-" not in json.dumps(payload)
    assert "raw-secret" not in json.dumps(payload)


def test_security_wrong_tenant_scope_activation_fails() -> None:
    revision_store, revision = _revision(scope=_SCOPE_A)
    service = _service(revision_store)
    with pytest.raises(EffectiveProfileActivationRevisionNotFoundError):
        _activate(service, revision, scope=_SCOPE_B, expected=None)


def test_active_binding_snapshot_is_coherent() -> None:
    revision_store, revision = _revision()
    active_store = InMemoryActiveEffectiveProfileRevisionStore()
    service = _service(revision_store, active_store)
    _activate(service, revision, expected=None)
    binding = active_store.get_active(_SCOPE_A)
    assert binding is not None
    resolved = resolve_active_effective_profile_revision(
        active_store=active_store,
        revision_store=revision_store,
        scope=_SCOPE_A,
    )
    assert binding.revision_id == resolved.revision_id
    assert binding.fingerprint == resolved.fingerprint


def test_kv_active_store_durable_adapter() -> None:
    backing = InMemoryKVStore()
    revision_store = InMemoryEffectiveProfileRevisionStore()
    _, revision = _revision(store=revision_store)
    active_store = KvActiveEffectiveProfileRevisionStore(backing)
    service = EffectiveProfileActivationService(
        EffectiveProfileActivationDependencies(
            revision_store=revision_store,
            active_store=active_store,
        ),
    )
    activate_materialized_revision(
        service,
        scope=_SCOPE_A,
        candidate_revision_id=revision.revision_id,
    )
    assert active_store.is_durable is True
    binding = active_store.get_active(_SCOPE_A)
    assert binding is not None
    assert binding.revision_id == revision.revision_id
