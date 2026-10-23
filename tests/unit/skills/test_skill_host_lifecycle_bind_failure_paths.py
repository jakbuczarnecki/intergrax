# © Artur Czarnecki. All rights reserved.

"""RC-03: Skill host bind failure paths — registry compensation and postconditions."""

from __future__ import annotations

import threading

import pytest

from intergrax.contracts.lifecycle_handoff.ack import DomainLifecycleHandoffDisposition
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityReleaseIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.skills.catalog import SkillCatalogProviderRegistry, SkillPackageResolution
from intergrax.skills.dynamic_acquisition import (
    DynamicSkillAcquisitionRequest,
    DynamicSkillAcquisitionService,
    SkillHostBindingMaterializer,
)
from intergrax.skills.errors import DynamicSkillAcquisitionBindingError
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.registry.profile import is_skill_enabled
from intergrax.skills.registry.provenance import SkillRuntimeBindingMetadata
from intergrax.skills.registry.runtime import SkillRegistry
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V1,
    me15_manifest_for_release,
)
from testing_support.me15_skill_binding_materializer import Me15SkillHostBindingMaterializer
from testing_support.me15_skill_catalog_provider import Me15SkillCatalogProvider
from tests.unit.skills.test_skill_host_lifecycle_batch_a import (
    _acquisition_stack,
    _candidate,
    _identity_key,
)

pytestmark = pytest.mark.unit

_OTHER_SKILL_ID = "skills.other.unexpected"


def _resolution(provider: Me15SkillCatalogProvider) -> SkillPackageResolution:
    entry = provider.list_entries()[0]
    return provider.resolve_package(entry, version_selector=ME15_VERSION_V1)


class _MutateThenRaiseMaterializer(SkillHostBindingMaterializer):
    def __init__(self, registry: SkillRegistry, *, catalog_source_id: str) -> None:
        self._inner = Me15SkillHostBindingMaterializer(registry, catalog_source_id=catalog_source_id)

    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        self._inner.materialize(resolution)
        raise RuntimeError("partial materialization")


class _WrongIdMaterializer(SkillHostBindingMaterializer):
    def __init__(self, registry: SkillRegistry, *, catalog_source_id: str) -> None:
        self._registry = registry
        self._catalog_source_id = catalog_source_id

    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        candidate = resolution.package_candidate
        if candidate.package_digest is None:
            raise ValueError("package digest required")
        release = CapabilityReleaseIdentity(
            discovery=CapabilityDiscoveryIdentity(
                kind=CapabilityKind.SKILL,
                source=CapabilitySourceIdentity(
                    source_id=self._catalog_source_id,
                    source_kind=CapabilitySourceKind.OFFICIAL,
                ),
                logical=CapabilityLogicalIdentity(
                    kind=CapabilityKind.SKILL,
                    logical_id=candidate.logical_skill_id,
                ),
            ),
            version_label=candidate.package_version,
            content_digest=candidate.package_digest,
            package_reference=candidate.package_reference,
        )
        manifest = me15_manifest_for_release(release).model_copy(update={"skill_id": _OTHER_SKILL_ID})
        binding = SkillRuntimeBindingMetadata(
            catalog_source_id=self._catalog_source_id,
            logical_skill_id=_OTHER_SKILL_ID,
            package_reference=candidate.package_reference,
            version_label=candidate.package_version,
            content_digest=candidate.package_digest,
        )
        self._registry.register(manifest, binding=binding)
        return _OTHER_SKILL_ID, binding


class _WrongIdThenRaiseMaterializer(_WrongIdMaterializer):
    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        super().materialize(resolution)
        raise RuntimeError("after wrong id")


class _GatedMaterializer(SkillHostBindingMaterializer):
    """Blocks in materialize until released — exercises lock-held critical section."""

    def __init__(
        self,
        registry: SkillRegistry,
        *,
        catalog_source_id: str,
        entered_barrier: threading.Barrier,
        hold: threading.Event,
    ) -> None:
        self._inner = Me15SkillHostBindingMaterializer(registry, catalog_source_id=catalog_source_id)
        self._entered_barrier = entered_barrier
        self._hold = hold

    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        self._entered_barrier.wait(timeout=10)
        if not self._hold.wait(timeout=10):
            raise TimeoutError("gated materializer hold timed out")
        return self._inner.materialize(resolution)


class _BindingIdentityMismatchMaterializer(SkillHostBindingMaterializer):
    def __init__(self, registry: SkillRegistry, *, catalog_source_id: str) -> None:
        self._inner = Me15SkillHostBindingMaterializer(registry, catalog_source_id=catalog_source_id)

    def materialize(
        self,
        resolution: SkillPackageResolution,
    ) -> tuple[str, SkillRuntimeBindingMetadata]:
        skill_id, binding = self._inner.materialize(resolution)
        mismatched = binding.model_copy(update={"content_digest": ME15_DIGEST_V2})
        return skill_id, mismatched


def test_materializer_mutates_registry_then_raises_leaves_prior_state() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-1")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    ack = lifecycle.bind(
        operation_id="op-mutate-raise",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_MutateThenRaiseMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert not lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert lifecycle.registry.skill_ids() == []
    assert "op-mutate-raise" not in lifecycle._handoff_operation_ids


def test_materializer_returns_unexpected_id_restores_registry() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-2")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    ack = lifecycle.bind(
        operation_id="op-wrong-id",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_WrongIdMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert not lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not lifecycle.is_bound(_OTHER_SKILL_ID)


def test_materializer_writes_unexpected_id_then_raises() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-3")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    ack = lifecycle.bind(
        operation_id="op-wrong-raise",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_WrongIdThenRaiseMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert lifecycle.registry.skill_ids() == []


def test_binding_metadata_identity_mismatch_rolls_back() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-4")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    ack = lifecycle.bind(
        operation_id="op-binding-mismatch",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_BindingIdentityMismatchMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert not lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)


def test_existing_valid_registration_survives_failed_bind_attempt() -> None:
    lifecycle, service, provider = _acquisition_stack(host_profile_id="host-rc03-5")
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-prior-ok",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    before_meta = lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
    resolution = _resolution(provider)
    ack = lifecycle.bind(
        operation_id="op-fail-after-prior",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_MutateThenRaiseMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID) == before_meta


def test_retry_after_failed_materialization_succeeds() -> None:
    lifecycle, service, provider = _acquisition_stack(host_profile_id="host-rc03-6")
    resolution = _resolution(provider)
    first = lifecycle.bind(
        operation_id="op-retry",
        host_profile_id=lifecycle.host_profile_id,
        resolved=resolution,
        materializer=_MutateThenRaiseMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert first.disposition is DomainLifecycleHandoffDisposition.REJECTED
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-retry",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)


def test_operation_id_not_replayable_as_success_after_failed_bind() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-7")
    provider = Me15SkillCatalogProvider()
    service = DynamicSkillAcquisitionService(
        catalog_registry=SkillCatalogProviderRegistry({provider.catalog_source_id: provider}),
        binding=lifecycle,
        materializer=_MutateThenRaiseMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    request = DynamicSkillAcquisitionRequest(
        operation_id="op-failed-once",
        host_profile_id=lifecycle.host_profile_id,
        capability_identity_key=_identity_key(provider),
        selected_identity=_candidate(provider=provider),
    )
    with pytest.raises(DynamicSkillAcquisitionBindingError):
        service.acquire(request)
    assert request.operation_id not in service._completed
    lifecycle._handoff_operation_ids.add(request.operation_id)
    replay = lifecycle.bind(
        operation_id=request.operation_id,
        host_profile_id=lifecycle.host_profile_id,
        resolved=_resolution(provider),
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert replay.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert replay.reason_detail == "idempotent handoff without bound skill"
    assert not lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)


def test_second_operation_cannot_overwrite_existing_valid_binding() -> None:
    lifecycle, service, provider = _acquisition_stack(host_profile_id="host-rc03-8")
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-first-bind",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    before = lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
    second = lifecycle.bind(
        operation_id="op-second-bind",
        host_profile_id=lifecycle.host_profile_id,
        resolved=_resolution(provider),
        materializer=Me15SkillHostBindingMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert second.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID) == before


def test_skill_stays_disabled_after_install_and_idempotent_replay() -> None:
    lifecycle, service, provider = _acquisition_stack(host_profile_id="host-rc03-9")
    request = DynamicSkillAcquisitionRequest(
        operation_id="op-disabled",
        host_profile_id=lifecycle.host_profile_id,
        capability_identity_key=_identity_key(provider),
        selected_identity=_candidate(provider=provider),
    )
    service.acquire(request)
    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not is_skill_enabled(lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)
    service.acquire(request)
    assert not is_skill_enabled(lifecycle.skill_profile, ME15_SKILL_LOGICAL_ID)


def test_cross_host_bind_failure_does_not_alter_peer_registry() -> None:
    life_a = SkillHostLifecycleService(host_profile_id="host-a-rc03")
    life_b = SkillHostLifecycleService(host_profile_id="host-b-rc03")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    ack = life_a.bind(
        operation_id="op-host-a-fail",
        host_profile_id=life_a.host_profile_id,
        resolved=resolution,
        materializer=_MutateThenRaiseMaterializer(
            life_a.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ack.disposition is DomainLifecycleHandoffDisposition.REJECTED
    assert not life_a.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not life_b.is_bound(ME15_SKILL_LOGICAL_ID)
    ok = life_b.bind(
        operation_id="op-host-b-ok",
        host_profile_id=life_b.host_profile_id,
        resolved=resolution,
        materializer=Me15SkillHostBindingMaterializer(
            life_b.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    assert ok.disposition is DomainLifecycleHandoffDisposition.ACCEPTED
    assert life_b.is_bound(ME15_SKILL_LOGICAL_ID)
    assert not life_a.is_bound(ME15_SKILL_LOGICAL_ID)


def test_concurrent_failed_binds_on_same_host_leave_registry_clean() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-concurrent-fail")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    materializer = _MutateThenRaiseMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )

    def _attempt_bind(operation_id: str) -> None:
        lifecycle.bind(
            operation_id=operation_id,
            host_profile_id=lifecycle.host_profile_id,
            resolved=resolution,
            materializer=materializer,
        )

    threads = [
        threading.Thread(target=_attempt_bind, args=(f"op-concurrent-fail-{index}",))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert lifecycle.registry.skill_ids() == []
    assert not lifecycle._handoff_operation_ids


def test_concurrent_distinct_operation_ids_serialize_to_single_install() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-concurrent-install")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    entered = threading.Barrier(2)
    hold = threading.Event()
    materializer = _GatedMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
        entered_barrier=entered,
        hold=hold,
    )
    results: list = []
    lock = threading.Lock()

    def _bind(operation_id: str) -> None:
        ack = lifecycle.bind(
            operation_id=operation_id,
            host_profile_id=lifecycle.host_profile_id,
            resolved=resolution,
            materializer=materializer,
        )
        with lock:
            results.append(ack)

    threads = [
        threading.Thread(target=_bind, args=("op-concurrent-a",)),
        threading.Thread(target=_bind, args=("op-concurrent-b",)),
    ]
    for thread in threads:
        thread.start()
    entered.wait(timeout=10)
    hold.set()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert len(lifecycle._handoff_operation_ids) == 1
    dispositions = {ack.disposition for ack in results}
    assert DomainLifecycleHandoffDisposition.ACCEPTED in dispositions
    assert DomainLifecycleHandoffDisposition.REJECTED in dispositions
    rejected = [ack for ack in results if ack.disposition is DomainLifecycleHandoffDisposition.REJECTED]
    assert rejected[0].reason_detail == "skill already bound for host profile"


def test_concurrent_replay_waits_for_in_progress_bind_commit() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="host-rc03-concurrent-replay")
    provider = Me15SkillCatalogProvider()
    resolution = _resolution(provider)
    entered = threading.Barrier(2)
    hold = threading.Event()
    materializer = _GatedMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
        entered_barrier=entered,
        hold=hold,
    )
    replay_ack: DomainLifecycleHandoffDisposition | None = None

    def _first_bind() -> None:
        lifecycle.bind(
            operation_id="op-same-replay",
            host_profile_id=lifecycle.host_profile_id,
            resolved=resolution,
            materializer=materializer,
        )

    def _replay_bind() -> None:
        nonlocal replay_ack
        ack = lifecycle.bind(
            operation_id="op-same-replay",
            host_profile_id=lifecycle.host_profile_id,
            resolved=resolution,
            materializer=Me15SkillHostBindingMaterializer(
                lifecycle.registry,
                catalog_source_id=provider.catalog_source_id,
            ),
        )
        replay_ack = ack.disposition

    first = threading.Thread(target=_first_bind)
    replay = threading.Thread(target=_replay_bind)
    first.start()
    replay.start()
    entered.wait(timeout=10)
    assert not lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    hold.set()
    first.join(timeout=10)
    replay.join(timeout=10)
    assert not first.is_alive()
    assert not replay.is_alive()

    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert replay_ack is DomainLifecycleHandoffDisposition.ACCEPTED
    assert lifecycle._handoff_operation_ids == {"op-same-replay"}


def test_concurrent_failed_bind_preserves_prior_successful_registration() -> None:
    lifecycle, service, provider = _acquisition_stack(host_profile_id="host-rc03-concurrent-prior")
    service.acquire(
        DynamicSkillAcquisitionRequest(
            operation_id="op-prior-concurrent",
            host_profile_id=lifecycle.host_profile_id,
            capability_identity_key=_identity_key(provider),
            selected_identity=_candidate(provider=provider),
        ),
    )
    before_meta = lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
    resolution = _resolution(provider)
    materializer = _MutateThenRaiseMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )

    def _attempt_fail(operation_id: str) -> None:
        lifecycle.bind(
            operation_id=operation_id,
            host_profile_id=lifecycle.host_profile_id,
            resolved=resolution,
            materializer=materializer,
        )

    threads = [
        threading.Thread(target=_attempt_fail, args=(f"op-fail-{index}",))
        for index in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert lifecycle.is_bound(ME15_SKILL_LOGICAL_ID)
    assert lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID) == before_meta
