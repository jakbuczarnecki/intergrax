# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Protocol, runtime_checkable

import pytest

from intergrax.agent_distribution.activation import (
    FakeRuntimeServingProjectionCoordinator,
)
from intergrax.agent_distribution.admin_models import (
    ActivateRuntimeRevisionRequest,
    BindAgentRequest,
    SetAgentEnablementRequest,
)
from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryRequest,
    AgentDiscoveryStrategyId,
    StaticAgentDiscoveryStrategy,
    project_package_contract_capabilities,
    project_to_capability_candidate,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.agent_distribution.agent_selection import (
    DeterministicIdentitySelectionStrategy,
    SelectionOutcome,
    build_agent_selection_request,
    require_selected_identity,
)
from intergrax.agent_distribution.capability_matching import CapabilityMatcher
from intergrax.agent_distribution.catalog import (
    AgentCatalogEntry,
    AgentDiscoveryCandidateIdentity,
    CatalogPackageResolution,
    CatalogProviderKind,
    CatalogSourceIdentity,
)
from intergrax.agent_distribution.dynamic_acquisition import (
    CatalogSourceProviderRegistry,
    DynamicAgentAcquisitionInstallIntent,
    DynamicAgentAcquisitionOutcome,
    DynamicAgentAcquisitionRequest,
    DynamicAgentAcquisitionResult,
    DynamicAgentAcquisitionService,
)
from intergrax.agent_distribution.federated_discovery import (
    FederatedAgentDiscoveryStrategy,
)
from intergrax.agent_distribution.identity import AgentPackageCandidate
from intergrax.agent_distribution.runtime_revision import RuntimeRevisionState
from intergrax.agent_distribution.task_capability_resolution import (
    build_deterministic_task_capability_resolver,
    build_task_capability_resolution_request,
    build_task_capability_rule,
)
from intergrax.agent_distribution.task_scoped_agents import (
    BindingTaskOrigin,
    BindingTaskOriginObservation,
    InMemoryTaskScopedAgentLeaseStore,
    TaskScopedAgentAcquisitionOutcome,
    TaskScopedAgentAcquisitionRequest,
    TaskScopedAgentAcquisitionService,
    TaskScopedAgentLease,
    TaskScopedAgentLeaseId,
    TaskScopedAgentLeaseNotFound,
    TaskScopedAgentLeaseConflict,
    TaskScopedAgentLeaseState,
    TaskScopedAgentOwnershipError,
    TaskScopedAgentReleaseError,
    TaskScopedAgentReleaseOutcome,
    TaskScopedAgentReleaseRequest,
    TaskScopedAgentService,
    TaskScopedOwnershipMode,
    binding_requires_runtime_release,
    finalize_binding_task_origin_authority,
)
from intergrax.contracts.execution_identity import TaskId, mint_task_id
from intergrax.core.qualification import QualificationStatus
from intergrax.agent_distribution.trust import (
    AgentInstallationTrustRecord,
    AgentQualificationEvidenceKind,
    AgentTrustEvidenceRef,
)
from tests.unit.agent_distribution.test_agent_platform_admin_service import (
    AdminStack,
    _APP,
    _ARTIFACT,
    _DIGEST,
    _ENV,
    _META_REF,
    _PACKAGE_ID,
    _bind_request,
    _build_request,
    _install_request,
    admin_test_principal,
    build_admin_stack,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SOURCE_ID = "builtin-1"
_CATALOG_ENTRY_ID = "cat-researcher"
_SLOT = "slot-search"
_INSTALL_ID = "inst-1"
_BINDING_ID = "bind-search"
_LOGICAL_AGENT = "researcher"
_TRANSLATION_BINDING = "bind-translate"
_TRANSLATION_SLOT = "slot-translate"
_TRANSLATION_INSTALL = "inst-translate"


def _source() -> CatalogSourceIdentity:
    return CatalogSourceIdentity(
        catalog_source_id=_SOURCE_ID,
        provider_kind=CatalogProviderKind.BUILTIN,
    )


def _candidate(*, digest: str | None = _DIGEST) -> AgentPackageCandidate:
    return AgentPackageCandidate(
        distribution_package_id=_PACKAGE_ID,
        package_version="1.0.0",
        package_digest=digest,
    )


def _identity(*, digest: str | None = _DIGEST) -> AgentDiscoveryCandidateIdentity:
    return AgentDiscoveryCandidateIdentity(
        source=_source(), package=_candidate(digest=digest)
    )


def _catalog_entry() -> AgentCatalogEntry:
    return AgentCatalogEntry(
        catalog_entry_id=_CATALOG_ENTRY_ID,
        catalog_source=_source(),
        display_name="Researcher",
        package_id_line=_PACKAGE_ID,
    )


def _resolution() -> CatalogPackageResolution:
    return CatalogPackageResolution(
        entry=_catalog_entry(),
        package_candidate=_candidate(),
        artifact_locator="catalog://artifact/researcher",
    )


class _ExactCatalog:
    def __init__(self) -> None:
        self._entry = _catalog_entry()
        self._resolution = _resolution()

    @property
    def catalog_source_id(self) -> str:
        return self._entry.catalog_source.catalog_source_id

    def list_entries(self, filters: object | None = None) -> list[AgentCatalogEntry]:
        del filters
        return [self._entry]

    def resolve_package(
        self,
        entry: AgentCatalogEntry,
        *,
        version_selector: str,
    ) -> CatalogPackageResolution:
        del entry, version_selector
        return self._resolution

    def health(self) -> None:
        return None


def _trust() -> AgentInstallationTrustRecord:
    return AgentInstallationTrustRecord(
        qualification_status=QualificationStatus.PRODUCTION_QUALIFIED,
        package_digest=_DIGEST,
        publisher_identity_ref="publisher:acme",
        source_provider_id="builtin",
        trust_evidence_refs=(
            AgentTrustEvidenceRef(
                evidence_id="evidence:service:0",
                kind=AgentQualificationEvidenceKind.SIGNATURE_VERIFICATION,
            ),
        ),
    )


def _activate_request(
    revision_id: str,
    *,
    pointer_revision: int = 0,
    prior_revision_id: str | None = None,
    mutation_id: str = "mut-activate",
) -> ActivateRuntimeRevisionRequest:
    return ActivateRuntimeRevisionRequest(
        mutation_id=mutation_id,
        runtime_revision_id=revision_id,
        artifact_locator="test://artifact",
        expected_artifact_digest=_ARTIFACT,
        expected_serving_pointer_revision=pointer_revision,
        expected_prior_traffic_revision_id=prior_revision_id,
    )


def _acquisition_request(
    revision_id: str,
    *,
    identity: AgentDiscoveryCandidateIdentity | None = None,
    install_mutation_id: str = "mut-install",
    bind_mutation_id: str = "mut-bind",
    pointer_revision: int = 0,
    prior_revision_id: str | None = None,
    binding_id: str = _BINDING_ID,
    install_id: str = _INSTALL_ID,
    slot_id: str = _SLOT,
) -> DynamicAgentAcquisitionRequest:
    return DynamicAgentAcquisitionRequest(
        selected_identity=identity or _identity(),
        application_id=_APP,
        application_environment_id=_ENV,
        catalog_entry_id=_CATALOG_ENTRY_ID,
        install=DynamicAgentAcquisitionInstallIntent(
            mutation_id=install_mutation_id,
            installation_id=install_id,
            installation_slot_id=slot_id,
            artifact_store_ref="store://artifacts/inst-1",
            trust_record=_trust(),
            agent_project_metadata_ref=_META_REF,
        ),
        bind=BindAgentRequest(
            mutation_id=bind_mutation_id,
            application_binding_id=binding_id,
            logical_agent_id=_LOGICAL_AGENT
            if binding_id == _BINDING_ID
            else "translator",
            installation_slot_id=slot_id,
            enablement=True,
        ),
        build=_build_request(revision_id, mutation_id=f"{install_mutation_id}-build"),
        activate=_activate_request(
            revision_id,
            pointer_revision=pointer_revision,
            prior_revision_id=prior_revision_id,
            mutation_id=f"{install_mutation_id}-activate",
        ),
    )


def _task_acquire_request(
    lease_id: str,
    task_scope_id: TaskId,
    revision_id: str,
    **kwargs: object,
) -> TaskScopedAgentAcquisitionRequest:
    return TaskScopedAgentAcquisitionRequest(
        lease_id=TaskScopedAgentLeaseId(lease_id),
        task_scope_id=task_scope_id,
        acquisition_request=_acquisition_request(revision_id, **kwargs),
    )


def _release_request(
    lease_id: str,
    task_scope_id: TaskId,
    revision_id: str,
    *,
    disable_revision: int = 1,
    pointer_revision: int = 1,
    prior_revision_id: str | None = None,
    disable_mutation_id: str = "mut-disable",
) -> TaskScopedAgentReleaseRequest:
    return TaskScopedAgentReleaseRequest(
        lease_id=TaskScopedAgentLeaseId(lease_id),
        task_scope_id=task_scope_id,
        application_id=_APP,
        application_environment_id=_ENV,
        disable=SetAgentEnablementRequest(
            mutation_id=disable_mutation_id,
            expected_revision=disable_revision,
        ),
        build=_build_request(revision_id, mutation_id=f"{disable_mutation_id}-build"),
        activate=_activate_request(
            revision_id,
            pointer_revision=pointer_revision,
            prior_revision_id=prior_revision_id,
            mutation_id=f"{disable_mutation_id}-activate",
        ),
    )


@dataclass
class TaskScopedHarness:
    stack: AdminStack
    service: TaskScopedAgentService
    lease_store: InMemoryTaskScopedAgentLeaseStore


def build_task_scoped_harness() -> TaskScopedHarness:
    stack = build_admin_stack(with_catalog=False)
    catalog = _ExactCatalog()
    registry = CatalogSourceProviderRegistry({_SOURCE_ID: catalog})
    acquisition = DynamicAgentAcquisitionService(
        catalog_registry=registry,
        lifecycle=stack.service,
    )
    lease_store = InMemoryTaskScopedAgentLeaseStore()
    service = TaskScopedAgentService(
        acquisition=acquisition,
        lifecycle=stack.service,
        lease_store=lease_store,
    )
    return TaskScopedHarness(stack=stack, service=service, lease_store=lease_store)


def _binding_revision(stack: AdminStack, binding_id: str = _BINDING_ID) -> int:
    binding = stack.service._binding_store.get_binding(binding_id)
    assert binding is not None
    return binding.binding_revision


def _agent_routable(
    stack: AdminStack, *, logical_agent_id: str = _LOGICAL_AGENT
) -> bool:
    status = stack.service.inspect_agent_status(
        application_id=_APP,
        application_environment_id=_ENV,
        logical_agent_id=logical_agent_id,
    )
    return (
        status.enabled_in_desired_state
        and status.included_in_active_revision
        and status.traffic_serving_revision_id is not None
    )


@runtime_checkable
class _LeaseStoreProtocol(Protocol):
    def get(self, lease_id: TaskScopedAgentLeaseId) -> object | None: ...

    def put_new(self, lease: object) -> None: ...

    def compare_and_set(
        self,
        lease_id: TaskScopedAgentLeaseId,
        *,
        expected_state: TaskScopedAgentLeaseState,
        new_lease: object,
    ) -> bool: ...

    def list_active_by_binding(
        self, application_binding_id: str
    ) -> tuple[object, ...]: ...

    def list_active_by_task_scope(
        self, task_scope_id: TaskId
    ) -> tuple[object, ...]: ...

    def get_binding_task_origin(self, application_binding_id: str) -> object | None: ...

    def reconcile_binding_task_origin(self, observation: object) -> object: ...

    def list_leases_by_binding(
        self, application_binding_id: str
    ) -> tuple[object, ...]: ...

    def finalize_binding_task_origin(self, application_binding_id: str) -> object: ...


def test_lease_store_protocol_is_structural() -> None:
    store = InMemoryTaskScopedAgentLeaseStore()
    assert isinstance(store, _LeaseStoreProtocol)


def test_acquire_task_scoped_lease_becomes_active() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    result = harness.service.acquire(
        _task_acquire_request("lease-1", task_scope, "rev-task-1"),
        principal=admin_test_principal(),
    )
    assert result.outcome is TaskScopedAgentAcquisitionOutcome.LEASE_ACQUIRED
    assert result.lease.lease_state is TaskScopedAgentLeaseState.ACTIVE
    assert result.lease.task_scope_id == task_scope
    assert _agent_routable(harness.stack)


def test_same_lease_reacquire_is_idempotent() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    request = _task_acquire_request("lease-1", task_scope, "rev-task-1")
    first = harness.service.acquire(request, principal=admin_test_principal())
    second = harness.service.acquire(request, principal=admin_test_principal())
    assert first.outcome is TaskScopedAgentAcquisitionOutcome.LEASE_ACQUIRED
    assert second.outcome is TaskScopedAgentAcquisitionOutcome.LEASE_REUSED
    assert second.lease == first.lease
    assert len(harness.lease_store.list_active_by_binding(_BINDING_ID)) == 1


def test_conflicting_same_lease_id_fails_closed() -> None:
    harness = build_task_scoped_harness()
    task_a = mint_task_id()
    task_b = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-1", task_a, "rev-task-1"),
        principal=admin_test_principal(),
    )
    with pytest.raises(TaskScopedAgentLeaseConflict, match="conflicting"):
        harness.service.acquire(
            _task_acquire_request("lease-1", task_b, "rev-task-2"),
            principal=admin_test_principal(),
        )


def test_two_task_leases_share_binding() -> None:
    harness = build_task_scoped_harness()
    task_a = mint_task_id()
    task_b = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-a", task_a, "rev-shared-1"),
        principal=admin_test_principal(),
    )
    harness.service.acquire(
        _task_acquire_request(
            "lease-b",
            task_b,
            "rev-shared-1",
            install_mutation_id="mut-install-b",
            bind_mutation_id="mut-bind-b",
        ),
        principal=admin_test_principal(),
    )
    active = harness.lease_store.list_active_by_binding(_BINDING_ID)
    assert len(active) == 2


def test_release_first_lease_retains_binding() -> None:
    harness = build_task_scoped_harness()
    task_a = mint_task_id()
    task_b = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-a", task_a, "rev-shared-1"),
        principal=admin_test_principal(),
    )
    harness.service.acquire(
        _task_acquire_request(
            "lease-b",
            task_b,
            "rev-shared-1",
            install_mutation_id="mut-install-b",
            bind_mutation_id="mut-bind-b",
        ),
        principal=admin_test_principal(),
    )
    released = harness.service.release(
        _release_request(
            "lease-a",
            task_a,
            "rev-release-a",
            prior_revision_id="rev-shared-1",
        ),
        principal=admin_test_principal(),
    )
    assert (
        released.outcome
        is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RETAINED_BINDING
    )
    assert _agent_routable(harness.stack)
    binding = harness.stack.service._binding_store.get_binding(_BINDING_ID)
    assert binding is not None
    assert binding.enablement is True


def test_release_final_lease_disables_binding() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-final", task_scope, "rev-active"),
        principal=admin_test_principal(),
    )
    released = harness.service.release(
        _release_request(
            "lease-final",
            task_scope,
            "rev-released",
            disable_revision=_binding_revision(harness.stack),
            prior_revision_id="rev-active",
        ),
        principal=admin_test_principal(),
    )
    assert (
        released.outcome is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED
    )
    binding = harness.stack.service._binding_store.get_binding(_BINDING_ID)
    assert binding is not None
    assert binding.enablement is False


def test_release_creates_canonical_runtime_revision() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-rev", task_scope, "rev-active"),
        principal=admin_test_principal(),
    )
    harness.service.release(
        _release_request(
            "lease-rev",
            task_scope,
            "rev-post-release",
            disable_revision=_binding_revision(harness.stack),
            prior_revision_id="rev-active",
        ),
        principal=admin_test_principal(),
    )
    revision = harness.stack.service._revision_store.get_revision("rev-post-release")
    assert revision is not None
    assert revision.revision_state is RuntimeRevisionState.ACTIVE


def test_released_specialist_no_longer_routable() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-route", task_scope, "rev-active"),
        principal=admin_test_principal(),
    )
    assert _agent_routable(harness.stack)
    harness.service.release(
        _release_request(
            "lease-route",
            task_scope,
            "rev-post-release",
            disable_revision=_binding_revision(harness.stack),
            prior_revision_id="rev-active",
        ),
        principal=admin_test_principal(),
    )
    assert not _agent_routable(harness.stack)


def _origin_observation(
    *,
    binding_id: str = _BINDING_ID,
    binding_created_by_task: bool,
) -> BindingTaskOriginObservation:
    return BindingTaskOriginObservation(
        application_binding_id=binding_id,
        binding_created_by_task=binding_created_by_task,
    )


def _lease_record(
    lease_id: str,
    task_scope_id: TaskId,
    *,
    binding_created_by_task: bool,
    binding_id: str = _BINDING_ID,
) -> TaskScopedAgentLease:
    return TaskScopedAgentLease(
        lease_id=TaskScopedAgentLeaseId(lease_id),
        task_scope_id=task_scope_id,
        application_id=_APP,
        application_environment_id=_ENV,
        ownership_mode=TaskScopedOwnershipMode.TASK_SCOPED,
        selected_identity=_identity(),
        installation_id=_INSTALL_ID,
        application_binding_id=binding_id,
        acquisition_runtime_revision_id="rev-1",
        binding_created_by_task=binding_created_by_task,
        lease_state=TaskScopedAgentLeaseState.ACTIVE,
    )


def test_reconcile_reused_before_created_resolves_task_created() -> None:
    store = InMemoryTaskScopedAgentLeaseStore()
    assert (
        store.reconcile_binding_task_origin(
            _origin_observation(binding_created_by_task=False),
        )
        is BindingTaskOrigin.UNRESOLVED
    )
    assert (
        store.reconcile_binding_task_origin(
            _origin_observation(binding_created_by_task=True),
        )
        is BindingTaskOrigin.TASK_CREATED
    )
    assert store.get_binding_task_origin(_BINDING_ID) is BindingTaskOrigin.TASK_CREATED


def test_reconcile_created_before_reused_resolves_task_created() -> None:
    store = InMemoryTaskScopedAgentLeaseStore()
    assert (
        store.reconcile_binding_task_origin(
            _origin_observation(binding_created_by_task=True),
        )
        is BindingTaskOrigin.TASK_CREATED
    )
    assert (
        store.reconcile_binding_task_origin(
            _origin_observation(binding_created_by_task=False),
        )
        is BindingTaskOrigin.TASK_CREATED
    )


def test_lease_insert_order_independence_for_binding_origin() -> None:
    task_a = mint_task_id()
    task_b = mint_task_id()
    lease_a = _lease_record("lease-a", task_a, binding_created_by_task=True)
    lease_b = _lease_record("lease-b", task_b, binding_created_by_task=False)

    store_ab = InMemoryTaskScopedAgentLeaseStore()
    store_ab.reconcile_binding_task_origin(
        _origin_observation(binding_created_by_task=False),
    )
    store_ab.reconcile_binding_task_origin(
        _origin_observation(binding_created_by_task=True),
    )
    store_ab.put_new(lease_a)
    store_ab.put_new(lease_b)

    store_ba = InMemoryTaskScopedAgentLeaseStore()
    store_ba.reconcile_binding_task_origin(
        _origin_observation(binding_created_by_task=False),
    )
    store_ba.reconcile_binding_task_origin(
        _origin_observation(binding_created_by_task=True),
    )
    store_ba.put_new(lease_b)
    store_ba.put_new(lease_a)

    for store in (store_ab, store_ba):
        assert (
            finalize_binding_task_origin_authority(
                lease_store=store,
                application_binding_id=_BINDING_ID,
            )
            is BindingTaskOrigin.TASK_CREATED
        )
        assert not binding_requires_runtime_release(
            lease_store=store,
            application_binding_id=_BINDING_ID,
            excluding_lease_id=TaskScopedAgentLeaseId("lease-a"),
        )
        assert (
            binding_requires_runtime_release(
                lease_store=store,
                application_binding_id=_BINDING_ID,
                excluding_lease_id=TaskScopedAgentLeaseId("lease-b"),
            )
            is False
        )
        store.compare_and_set(
            TaskScopedAgentLeaseId("lease-b"),
            expected_state=TaskScopedAgentLeaseState.ACTIVE,
            new_lease=lease_b.model_copy(
                update={"lease_state": TaskScopedAgentLeaseState.RELEASED},
            ),
        )
        assert binding_requires_runtime_release(
            lease_store=store,
            application_binding_id=_BINDING_ID,
            excluding_lease_id=TaskScopedAgentLeaseId("lease-a"),
        )


def test_true_pre_existing_binding_origin_finalizes_pre_existing() -> None:
    store = InMemoryTaskScopedAgentLeaseStore()
    task_scope = mint_task_id()
    store.reconcile_binding_task_origin(
        _origin_observation(binding_created_by_task=False),
    )
    store.put_new(
        _lease_record("lease-persistent", task_scope, binding_created_by_task=False),
    )
    assert (
        finalize_binding_task_origin_authority(
            lease_store=store,
            application_binding_id=_BINDING_ID,
        )
        is BindingTaskOrigin.PRE_EXISTING
    )
    assert not binding_requires_runtime_release(
        lease_store=store,
        application_binding_id=_BINDING_ID,
        excluding_lease_id=TaskScopedAgentLeaseId("lease-persistent"),
    )


def test_contradictory_pre_existing_and_created_observations_fail_closed() -> None:
    store = InMemoryTaskScopedAgentLeaseStore()
    store._binding_origins[_BINDING_ID] = BindingTaskOrigin.PRE_EXISTING
    with pytest.raises(TaskScopedAgentOwnershipError, match="contradictory"):
        store.reconcile_binding_task_origin(
            _origin_observation(binding_created_by_task=True),
        )


def test_unresolved_origin_without_lease_evidence_fails_closed() -> None:
    store = InMemoryTaskScopedAgentLeaseStore()
    store._binding_origins[_BINDING_ID] = BindingTaskOrigin.UNRESOLVED
    with pytest.raises(TaskScopedAgentOwnershipError, match="unresolved"):
        store.finalize_binding_task_origin(_BINDING_ID)


class _InterleavingAcquisition:
    """Deterministic first-call pause between Phase 6 and origin reconciliation."""

    def __init__(self, inner: DynamicAgentAcquisitionService) -> None:
        self._inner = inner
        self._call_count = 0
        self._first_paused = Event()
        self._first_resume = Event()

    def acquire(
        self,
        request: DynamicAgentAcquisitionRequest,
        *,
        principal: object,
    ) -> DynamicAgentAcquisitionResult:
        self._call_count += 1
        call_index = self._call_count
        result = self._inner.acquire(request, principal=principal)
        if call_index == 1:
            self._first_paused.set()
            assert self._first_resume.wait(timeout=5)
            return result.model_copy(update={"binding_reused": False})
        return result.model_copy(update={"binding_reused": True})


def test_concurrent_acquisition_reused_before_created_disables_on_final_release() -> (
    None
):
    harness = build_task_scoped_harness()
    task_a = mint_task_id()
    task_b = mint_task_id()
    interleaving = _InterleavingAcquisition(
        DynamicAgentAcquisitionService(
            catalog_registry=CatalogSourceProviderRegistry(
                {_SOURCE_ID: _ExactCatalog()},
            ),
            lifecycle=harness.stack.service,
        ),
    )
    acquisition_service = TaskScopedAgentAcquisitionService(
        acquisition=interleaving,
        lease_store=harness.lease_store,
    )

    errors: list[BaseException] = []

    def acquire_a() -> None:
        try:
            acquisition_service.acquire(
                _task_acquire_request("lease-a", task_a, "rev-race-1"),
                principal=admin_test_principal(),
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    thread_a = Thread(target=acquire_a)
    thread_a.start()
    assert interleaving._first_paused.wait(timeout=5)

    acquisition_service.acquire(
        _task_acquire_request(
            "lease-b",
            task_b,
            "rev-race-1",
            install_mutation_id="mut-install-b",
            bind_mutation_id="mut-bind-b",
        ),
        principal=admin_test_principal(),
    )
    interleaving._first_resume.set()
    thread_a.join(timeout=5)
    assert not errors
    assert thread_a.is_alive() is False

    assert (
        harness.lease_store.get_binding_task_origin(_BINDING_ID)
        is BindingTaskOrigin.TASK_CREATED
    )

    first = harness.service.release(
        _release_request(
            "lease-a",
            task_a,
            "rev-race-release-a",
            prior_revision_id="rev-race-1",
        ),
        principal=admin_test_principal(),
    )
    assert (
        first.outcome is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RETAINED_BINDING
    )
    assert _agent_routable(harness.stack)

    second = harness.service.release(
        _release_request(
            "lease-b",
            task_b,
            "rev-race-release-b",
            disable_revision=_binding_revision(harness.stack),
            prior_revision_id="rev-race-1",
            pointer_revision=1,
        ),
        principal=admin_test_principal(),
    )
    assert (
        second.outcome is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED
    )
    assert not _agent_routable(harness.stack)


def test_persistent_binding_survives_task_release() -> None:
    harness = build_task_scoped_harness()
    principal = admin_test_principal()
    harness.stack.service.install_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_install_request(),
        principal=principal,
    )
    harness.stack.service.bind_agent(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_bind_request(),
        principal=principal,
    )
    harness.stack.service.enable_binding(
        application_id=_APP,
        application_environment_id=_ENV,
        application_binding_id=_BINDING_ID,
        request=SetAgentEnablementRequest(
            mutation_id="mut-enable-persistent",
            expected_revision=0,
        ),
        principal=principal,
    )
    harness.stack.service.build_application_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_build_request("rev-persistent"),
        principal=principal,
    )
    harness.stack.service.activate_revision(
        application_id=_APP,
        application_environment_id=_ENV,
        request=_activate_request("rev-persistent"),
        principal=principal,
    )
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request(
            "lease-persistent",
            task_scope,
            "rev-persistent",
            install_mutation_id="mut-install-task",
            bind_mutation_id="mut-bind-task",
        ),
        principal=principal,
    )
    released = harness.service.release(
        _release_request(
            "lease-persistent",
            task_scope,
            "rev-task-release",
            prior_revision_id="rev-persistent",
        ),
        principal=admin_test_principal(),
    )
    assert (
        released.outcome
        is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RETAINED_BINDING
    )
    assert _agent_routable(harness.stack)
    serving = harness.stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert serving.traffic_serving_revision_id == "rev-persistent"


def test_already_released_lease_is_idempotent() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-done", task_scope, "rev-active"),
        principal=admin_test_principal(),
    )
    request = _release_request(
        "lease-done",
        task_scope,
        "rev-released",
        disable_revision=_binding_revision(harness.stack),
        prior_revision_id="rev-active",
    )
    first = harness.service.release(request, principal=admin_test_principal())
    second = harness.service.release(request, principal=admin_test_principal())
    assert first.outcome is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED
    assert second.outcome is TaskScopedAgentReleaseOutcome.ALREADY_RELEASED


def test_wrong_task_scope_release_fails_closed() -> None:
    harness = build_task_scoped_harness()
    owner = mint_task_id()
    intruder = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-owner", owner, "rev-active"),
        principal=admin_test_principal(),
    )
    with pytest.raises(TaskScopedAgentOwnershipError, match="task_scope_id"):
        harness.service.release(
            _release_request(
                "lease-owner",
                intruder,
                "rev-released",
                disable_revision=_binding_revision(harness.stack),
                prior_revision_id="rev-active",
            ),
            principal=admin_test_principal(),
        )


def test_failed_prepare_preserves_active_revision() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-fail", task_scope, "rev-active"),
        principal=admin_test_principal(),
    )
    coordinator = FakeRuntimeServingProjectionCoordinator()
    coordinator.fail_prepare("rev-release-fail")
    harness.stack.service._activation_service._projection_coordinator = coordinator
    with pytest.raises(TaskScopedAgentReleaseError):
        harness.service.release(
            _release_request(
                "lease-fail",
                task_scope,
                "rev-release-fail",
                disable_revision=_binding_revision(harness.stack),
                prior_revision_id="rev-active",
            ),
            principal=admin_test_principal(),
        )
    serving = harness.stack.service.inspect_serving(
        application_id=_APP,
        application_environment_id=_ENV,
    )
    assert serving.traffic_serving_revision_id == "rev-active"
    lease = harness.lease_store.get(TaskScopedAgentLeaseId("lease-fail"))
    assert lease is not None
    assert lease.lease_state is TaskScopedAgentLeaseState.RELEASE_FAILED


def test_retry_release_after_failure_succeeds() -> None:
    harness = build_task_scoped_harness()
    task_scope = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-retry", task_scope, "rev-active"),
        principal=admin_test_principal(),
    )
    coordinator = FakeRuntimeServingProjectionCoordinator()
    coordinator.fail_prepare("rev-release-fail")
    harness.stack.service._activation_service._projection_coordinator = coordinator
    with pytest.raises(TaskScopedAgentReleaseError):
        harness.service.release(
            _release_request(
                "lease-retry",
                task_scope,
                "rev-release-fail",
                disable_revision=_binding_revision(harness.stack),
                prior_revision_id="rev-active",
            ),
            principal=admin_test_principal(),
        )
    harness.stack.service._activation_service._projection_coordinator = (
        FakeRuntimeServingProjectionCoordinator()
    )
    disable_revision = _binding_revision(harness.stack)
    result = harness.service.release(
        _release_request(
            "lease-retry",
            task_scope,
            "rev-release-retry",
            disable_revision=disable_revision,
            prior_revision_id="rev-active",
            pointer_revision=1,
            disable_mutation_id="mut-disable-retry",
        ),
        principal=admin_test_principal(),
    )
    assert (
        result.outcome is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED
    )
    assert result.lease.lease_state is TaskScopedAgentLeaseState.RELEASED


def test_reacquire_after_release_reuses_installation() -> None:
    harness = build_task_scoped_harness()
    task_a = mint_task_id()
    task_b = mint_task_id()
    first = harness.service.acquire(
        _task_acquire_request("lease-a", task_a, "rev-cycle-1"),
        principal=admin_test_principal(),
    )
    harness.service.release(
        _release_request(
            "lease-a",
            task_a,
            "rev-cycle-2",
            disable_revision=_binding_revision(harness.stack),
            prior_revision_id="rev-cycle-1",
        ),
        principal=admin_test_principal(),
    )
    second = harness.service.acquire(
        _task_acquire_request(
            "lease-b",
            task_b,
            "rev-cycle-3",
            install_mutation_id="mut-install-reacquire",
            bind_mutation_id="mut-bind-reacquire",
            pointer_revision=2,
            prior_revision_id="rev-cycle-2",
        ),
        principal=admin_test_principal(),
    )
    assert second.acquisition_result.installation_reused is True
    assert (
        second.acquisition_result.outcome
        is DynamicAgentAcquisitionOutcome.ACQUIRED_ACTIVE
    )
    assert _agent_routable(harness.stack)
    assert first.lease.installation_id == second.lease.installation_id


def test_release_unknown_lease_fails_closed() -> None:
    harness = build_task_scoped_harness()
    with pytest.raises(TaskScopedAgentLeaseNotFound):
        harness.service.release(
            _release_request(
                "lease-missing",
                mint_task_id(),
                "rev-missing",
            ),
            principal=admin_test_principal(),
        )


def test_two_different_agents_release_isolation() -> None:
    harness = build_task_scoped_harness()
    task_ocr = mint_task_id()
    task_translate = mint_task_id()
    harness.service.acquire(
        _task_acquire_request("lease-ocr", task_ocr, "rev-both-1"),
        principal=admin_test_principal(),
    )
    harness.service.acquire(
        _task_acquire_request(
            "lease-translate",
            task_translate,
            "rev-both-2",
            binding_id=_TRANSLATION_BINDING,
            install_id=_TRANSLATION_INSTALL,
            slot_id=_TRANSLATION_SLOT,
            install_mutation_id="mut-install-translate",
            bind_mutation_id="mut-bind-translate",
            pointer_revision=1,
            prior_revision_id="rev-both-1",
        ),
        principal=admin_test_principal(),
    )
    harness.service.release(
        _release_request(
            "lease-ocr",
            task_ocr,
            "rev-ocr-release",
            disable_revision=_binding_revision(harness.stack, _BINDING_ID),
            prior_revision_id="rev-both-2",
            pointer_revision=2,
        ),
        principal=admin_test_principal(),
    )
    translate_binding = harness.stack.service._binding_store.get_binding(
        _TRANSLATION_BINDING,
    )
    assert translate_binding is not None
    assert translate_binding.enablement is True


def test_task_scoped_module_has_no_registry_or_nexus_coupling() -> None:
    import intergrax.agent_distribution.task_scoped_agents as module

    source_path = Path(module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden_prefixes = (
        "intergrax.harness",
        "intergrax.nexus",
        "agents",
        "applications",
    )
    violations = sorted(
        imported
        for imported in imported_modules
        if any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert not violations, f"forbidden imports: {violations}"


def _discovery_candidate(capability_ids: tuple[str, ...]) -> AgentDiscoveryCandidate:
    return AgentDiscoveryCandidate(
        identity=_identity(),
        capabilities=project_package_contract_capabilities(
            AgentPackageContractDeclaration(
                contract_id="contract.v1",
                contract_version="1",
                capabilities=capability_ids,
            ),
        ),
        catalog_entry_id=_CATALOG_ENTRY_ID,
    )


def test_full_task_pipeline_acquire_lease_and_release() -> None:
    harness = build_task_scoped_harness()
    resolver = build_deterministic_task_capability_resolver(
        rules=(
            build_task_capability_rule(
                rule_id="rule.document.ocr.v1",
                task_kind="document.ocr",
                required=("document.ocr",),
                rule_version="1",
            ),
        ),
    )
    resolution = resolver.resolve(
        build_task_capability_resolution_request(task_kind="document.ocr"),
    )
    discovery = FederatedAgentDiscoveryStrategy(
        strategies=(
            StaticAgentDiscoveryStrategy(
                strategy_id=AgentDiscoveryStrategyId(value="static.test"),
                candidates=(_discovery_candidate(("document.ocr",)),),
            ),
        ),
    )
    discovered = discovery.discover(
        AgentDiscoveryRequest(requirement=resolution.capability_requirement),
    )
    matcher = CapabilityMatcher()
    match = matcher.match(
        requirement=resolution.capability_requirement,
        candidate=project_to_capability_candidate(discovered.candidates[0]),
    )
    decision = DeterministicIdentitySelectionStrategy().select(
        build_agent_selection_request(
            requirement=resolution.capability_requirement,
            eligible_matches=(match,) if match.eligible else (),
        ),
    )
    assert decision.outcome is SelectionOutcome.SELECTED
    selected = require_selected_identity(decision)
    task_scope = mint_task_id()
    acquired = harness.service.acquire(
        TaskScopedAgentAcquisitionRequest(
            lease_id=TaskScopedAgentLeaseId("lease-pipeline"),
            task_scope_id=task_scope,
            acquisition_request=_acquisition_request(
                "rev-pipeline-1",
                identity=selected,
            ),
        ),
        principal=admin_test_principal(),
    )
    assert acquired.lease.lease_state is TaskScopedAgentLeaseState.ACTIVE
    assert _agent_routable(harness.stack)
    released = harness.service.release(
        _release_request(
            "lease-pipeline",
            task_scope,
            "rev-pipeline-2",
            disable_revision=_binding_revision(harness.stack),
            prior_revision_id="rev-pipeline-1",
        ),
        principal=admin_test_principal(),
    )
    assert (
        released.outcome is TaskScopedAgentReleaseOutcome.LEASE_RELEASED_RUNTIME_UPDATED
    )
    assert not _agent_routable(harness.stack)
