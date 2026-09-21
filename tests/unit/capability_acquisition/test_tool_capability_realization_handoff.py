# © Artur Czarnecki. All rights reserved.

"""UCA-2 — Tool domain realization handoff and availability projection."""

from __future__ import annotations

from datetime import UTC, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pytest
from pydantic import ValidationError

from intergrax.capability_acquisition.adapters.tool import (
    ToolCapabilityRealizationProvider,
)
from intergrax.capability_acquisition.availability_projection import (
    project_availability_disposition,
)
from intergrax.capability_acquisition.service import CapabilityRealizationService
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    CapabilityRealizationReasonCode,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
    derive_capability_realization_request_id,
)
from intergrax.contracts.capability_catalog.availability import AvailabilityDisposition
from intergrax.contracts.capability_catalog.capability_realization_need import (
    CapabilityRealizationNeed,
)
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from intergrax.contracts.capability_catalog.identity import CapabilitySourceKind
from intergrax.contracts.capability_catalog.kind import CapabilityKind
from intergrax.contracts.lifecycle_handoff.ack import (
    DomainLifecycleHandoffAck,
    DomainLifecycleHandoffDisposition,
)
from intergrax.contracts.tools.known_capability_realization import (
    KnownToolCapabilityRealizationRequest,
    KnownToolCapabilityRealizationResult,
)
from intergrax.tools.catalog import (
    ToolCatalogEntry,
    ToolPackageCandidate,
    ToolPackageResolution,
)
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from intergrax.tools.errors import KnownToolCapabilityRealizationConflictError
from intergrax.tools.known_capability_realization import (
    ToolKnownCapabilityRealizationService,
)
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
)
from testing_support.me14_tool_activation_materializer import (
    Me14ToolHostActivationMaterializer,
)
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 10, 0, tzinfo=UTC)


def _need() -> CapabilityRealizationNeed:
    provider = Me14ToolCatalogProvider()
    key = CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id=provider.catalog_source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=ME14_TOOL_LOGICAL_ID,
    )
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(key,),
        created_at=_CREATED,
    )
    return CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=key,
    )


def _request(need: CapabilityRealizationNeed) -> CapabilityRealizationRequest:
    return CapabilityRealizationRequest(
        request_id=derive_capability_realization_request_id(
            realization_need_id=need.realization_need_id,
            request_nonce="nonce-1",
        ),
        request_nonce="nonce-1",
        realization_need=need,
        host_profile_id="host-profile-1",
        requested_at=_CREATED,
    )


class _StaticResolver:
    def __init__(self, resolution: ToolPackageResolution) -> None:
        self._resolution = resolution
        self.calls = 0

    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution:
        self.calls += 1
        return self._resolution


def _resolution() -> ToolPackageResolution:
    provider = Me14ToolCatalogProvider()
    entry = ToolCatalogEntry(
        catalog_entry_id="entry-me14",
        catalog_source_id=provider.catalog_source_id,
        logical_tool_id=ME14_TOOL_LOGICAL_ID,
        package_reference=ME14_PACKAGE_REFERENCE_V1,
        display_name="ME14 Echo Tool",
    )
    candidate = ToolPackageCandidate(
        logical_tool_id=ME14_TOOL_LOGICAL_ID,
        package_reference=ME14_PACKAGE_REFERENCE_V1,
        package_version=ME14_VERSION_V1,
        package_digest=ME14_DIGEST_V1,
    )
    return ToolPackageResolution(entry=entry, package_candidate=candidate)


def test_tool_handoff_success_projects_host_available() -> None:
    need = _need()
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    resolver = _StaticResolver(_resolution())
    domain = ToolKnownCapabilityRealizationService(
        activation=lifecycle,
        materializer=materializer,
        resolver=resolver,
    )
    provider = ToolCapabilityRealizationProvider(domain)
    service = CapabilityRealizationService((provider,))
    result = service.realize(_request(need))
    assert result.outcome is CapabilityRealizationOutcome.SUCCEEDED
    assert result.evidence is not None
    disposition = project_availability_disposition(
        identity=need.capability_identity,
        evidence=result.evidence,
    )
    assert disposition is AvailabilityDisposition.HOST_AVAILABLE


def test_tool_domain_idempotent_operation_id() -> None:
    need = _need()
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    resolver = _StaticResolver(_resolution())
    domain = ToolKnownCapabilityRealizationService(
        activation=lifecycle,
        materializer=materializer,
        resolver=resolver,
    )
    domain_request = KnownToolCapabilityRealizationRequest(
        operation_id="op-1",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    first = domain.realize(domain_request)
    second = domain.realize(domain_request)
    assert first is second
    assert resolver.calls == 1


def test_tool_request_rejects_non_tool_kind() -> None:
    need = _need()
    agent_key = CapabilityIdentityKey(
        kind=CapabilityKind.AGENT,
        source_id=need.capability_identity.source_id,
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=ME14_TOOL_LOGICAL_ID,
    )
    with pytest.raises(ValidationError):
        KnownToolCapabilityRealizationRequest(
            operation_id="op-1",
            host_profile_id="host-profile-1",
            capability_identity=agent_key,
            requested_at=_CREATED,
        )


class _CountingActivation:
    host_profile_id = "host-profile-1"

    def __init__(self, inner: ToolHostLifecycleService) -> None:
        self._inner = inner
        self.activate_calls = 0

    def registry_read(self):
        return self._inner.registry_read()

    def is_active(self, logical_tool_id: str) -> bool:
        return self._inner.is_active(logical_tool_id)

    def activation_metadata(self, logical_tool_id: str):
        return self._inner.activation_metadata(logical_tool_id)

    def activate(self, **kwargs):
        self.activate_calls += 1
        return self._inner.activate(**kwargs)


def _domain_with_resolver(
    resolution: ToolPackageResolution,
) -> tuple[ToolKnownCapabilityRealizationService, _CountingActivation]:
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    activation = _CountingActivation(lifecycle)
    resolver = _StaticResolver(resolution)
    domain = ToolKnownCapabilityRealizationService(
        activation=activation,
        materializer=materializer,
        resolver=resolver,
    )
    return domain, activation


def test_resolver_source_mismatch_fails_before_activation() -> None:
    need = _need()
    resolution = _resolution()
    bad_entry = resolution.entry.model_copy(
        update={"catalog_source_id": "wrong.source"},
    )
    bad = ToolPackageResolution(
        entry=bad_entry,
        package_candidate=resolution.package_candidate,
    )
    domain, activation = _domain_with_resolver(bad)
    result = domain.realize(
        KnownToolCapabilityRealizationRequest(
            operation_id="op-mismatch-source",
            host_profile_id="host-profile-1",
            capability_identity=need.capability_identity,
            requested_at=_CREATED,
        ),
    )
    assert result.outcome.name == "FAILED"
    assert activation.activate_calls == 0


def test_resolver_logical_id_mismatch_fails_before_activation() -> None:
    need = _need()
    resolution = _resolution()
    bad_candidate = resolution.package_candidate.model_copy(
        update={"logical_tool_id": "other.logical"},
    )
    bad = ToolPackageResolution(entry=resolution.entry, package_candidate=bad_candidate)
    domain, activation = _domain_with_resolver(bad)
    result = domain.realize(
        KnownToolCapabilityRealizationRequest(
            operation_id="op-mismatch-logical",
            host_profile_id="host-profile-1",
            capability_identity=need.capability_identity,
            requested_at=_CREATED,
        ),
    )
    assert result.outcome.name == "FAILED"
    assert activation.activate_calls == 0


def test_resolution_internal_entry_candidate_mismatch_fails() -> None:
    need = _need()
    resolution = _resolution()
    bad_entry = resolution.entry.model_copy(update={"logical_tool_id": "entry.other"})
    bad = ToolPackageResolution(
        entry=bad_entry,
        package_candidate=resolution.package_candidate,
    )
    domain, activation = _domain_with_resolver(bad)
    result = domain.realize(
        KnownToolCapabilityRealizationRequest(
            operation_id="op-internal",
            host_profile_id="host-profile-1",
            capability_identity=need.capability_identity,
            requested_at=_CREATED,
        ),
    )
    assert result.outcome.name == "FAILED"
    assert activation.activate_calls == 0


def test_resolution_package_reference_mismatch_fails() -> None:
    need = _need()
    resolution = _resolution()
    bad_candidate = resolution.package_candidate.model_copy(
        update={"package_reference": "pkg://other"},
    )
    bad = ToolPackageResolution(entry=resolution.entry, package_candidate=bad_candidate)
    domain, activation = _domain_with_resolver(bad)
    result = domain.realize(
        KnownToolCapabilityRealizationRequest(
            operation_id="op-pkg-ref",
            host_profile_id="host-profile-1",
            capability_identity=need.capability_identity,
            requested_at=_CREATED,
        ),
    )
    assert result.outcome.name == "FAILED"
    assert activation.activate_calls == 0


def test_replay_same_operation_id_different_identity_conflicts() -> None:
    need = _need()
    domain, _activation = _domain_with_resolver(_resolution())
    base = KnownToolCapabilityRealizationRequest(
        operation_id="op-replay-conflict",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    domain.realize(base)
    other_key = need.capability_identity.model_copy(update={"logical_id": "other.tool"})
    conflict = base.model_copy(update={"capability_identity": other_key})
    with pytest.raises(KnownToolCapabilityRealizationConflictError):
        domain.realize(conflict)


def test_replay_same_operation_id_different_host_conflicts() -> None:
    need = _need()
    domain, _activation = _domain_with_resolver(_resolution())
    base = KnownToolCapabilityRealizationRequest(
        operation_id="op-replay-host",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    domain.realize(base)
    conflict = base.model_copy(update={"host_profile_id": "host-profile-2"})
    with pytest.raises(KnownToolCapabilityRealizationConflictError):
        domain.realize(conflict)


def test_idempotent_replay_calls_activation_once() -> None:
    need = _need()
    domain, activation = _domain_with_resolver(_resolution())
    request = KnownToolCapabilityRealizationRequest(
        operation_id="op-idem",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    first = domain.realize(request)
    second = domain.realize(request)
    assert first.outcome == second.outcome
    assert activation.activate_calls == 1


class _FailingResolver:
    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution:
        raise LookupError("resolver failed")


class _FlakyResolver:
    def __init__(self, resolution: ToolPackageResolution) -> None:
        self._resolution = resolution
        self.calls = 0

    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution:
        self.calls += 1
        if self.calls == 1:
            raise LookupError("resolver failed")
        return self._resolution


def _domain_with_custom_resolver(
    resolver: object,
) -> tuple[ToolKnownCapabilityRealizationService, _CountingActivation]:
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    activation = _CountingActivation(lifecycle)
    domain = ToolKnownCapabilityRealizationService(
        activation=activation,
        materializer=materializer,
        resolver=resolver,
    )
    return domain, activation


def test_failed_first_attempt_then_conflicting_identity_conflicts() -> None:
    need = _need()
    domain, activation = _domain_with_custom_resolver(_FailingResolver())
    base = KnownToolCapabilityRealizationRequest(
        operation_id="op-1",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    failed = domain.realize(base)
    assert failed.outcome.name == "FAILED"
    assert activation.activate_calls == 0
    other_key = need.capability_identity.model_copy(update={"logical_id": "other.tool"})
    conflict = base.model_copy(update={"capability_identity": other_key})
    with pytest.raises(KnownToolCapabilityRealizationConflictError):
        domain.realize(conflict)


def test_failed_first_attempt_same_request_retry_allowed() -> None:
    need = _need()
    flaky = _FlakyResolver(_resolution())
    domain, activation = _domain_with_custom_resolver(flaky)
    request = KnownToolCapabilityRealizationRequest(
        operation_id="op-retry",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    first = domain.realize(request)
    assert first.outcome.name == "FAILED"
    second = domain.realize(request)
    assert second.outcome.name in {"REALIZED", "ALREADY_REALIZED"}
    assert activation.activate_calls == 1
    assert flaky.calls == 2


def test_different_host_after_failure_conflicts() -> None:
    need = _need()
    domain, activation = _domain_with_custom_resolver(_FailingResolver())
    base = KnownToolCapabilityRealizationRequest(
        operation_id="op-host-fail",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    failed = domain.realize(base)
    assert failed.outcome.name == "FAILED"
    assert activation.activate_calls == 0
    conflict = base.model_copy(update={"host_profile_id": "host-profile-2"})
    with pytest.raises(KnownToolCapabilityRealizationConflictError):
        domain.realize(conflict)


class _ConflictHandoff:
    def realize(
        self,
        request: KnownToolCapabilityRealizationRequest,
    ) -> None:
        raise KnownToolCapabilityRealizationConflictError("operation replay conflict")


def test_tool_adapter_maps_replay_conflict_to_uca_conflict() -> None:
    need = _need()
    provider = ToolCapabilityRealizationProvider(_ConflictHandoff())
    result = provider.realize(_request(need))
    assert result.outcome is CapabilityRealizationOutcome.CONFLICT
    assert (
        result.reason_code is CapabilityRealizationReasonCode.OPERATION_REPLAY_CONFLICT
    )


class _BlockingActivationGate:
    host_profile_id = "host-profile-1"

    def __init__(self, inner: ToolHostLifecycleService) -> None:
        self._inner = inner
        self.activate_calls = 0
        self.in_activate = threading.Event()
        self.release_activate = threading.Event()

    def registry_read(self):
        return self._inner.registry_read()

    def is_active(self, logical_tool_id: str) -> bool:
        return self._inner.is_active(logical_tool_id)

    def activation_metadata(self, logical_tool_id: str):
        return self._inner.activation_metadata(logical_tool_id)

    def activate(self, **kwargs):
        self.activate_calls += 1
        self.in_activate.set()
        self.release_activate.wait(timeout=5)
        return self._inner.activate(**kwargs)


def test_concurrent_identical_request_single_side_effect_path() -> None:
    need = _need()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    activation = _BlockingActivationGate(lifecycle)
    resolver = _StaticResolver(_resolution())
    provider = Me14ToolCatalogProvider()
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    domain = ToolKnownCapabilityRealizationService(
        activation=activation,
        materializer=materializer,
        resolver=resolver,
    )
    request = KnownToolCapabilityRealizationRequest(
        operation_id="op-concurrent-identical",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    results: list[KnownToolCapabilityRealizationResult] = []
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            results.append(domain.realize(request))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for thread in threads:
        thread.start()
    activation.in_activate.wait(timeout=5)
    assert activation.activate_calls == 1
    assert resolver.calls == 1
    activation.release_activate.set()
    for thread in threads:
        thread.join(timeout=5)
    assert not errors
    assert len(results) == 2
    assert results[0].outcome == results[1].outcome
    assert activation.activate_calls == 1
    assert resolver.calls == 1


def test_concurrent_conflicting_identity_one_binding() -> None:
    need = _need()
    domain, activation = _domain_with_resolver(_resolution())
    other_key = need.capability_identity.model_copy(update={"logical_id": "other.tool"})
    start = threading.Barrier(2)
    outcomes: list[object] = []

    def _run(request: KnownToolCapabilityRealizationRequest) -> None:
        start.wait(timeout=5)
        try:
            outcomes.append(domain.realize(request))
        except KnownToolCapabilityRealizationConflictError as exc:
            outcomes.append(exc)

    req_a = KnownToolCapabilityRealizationRequest(
        operation_id="op-conflict-id",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    req_b = req_a.model_copy(update={"capability_identity": other_key})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_run, req) for req in (req_a, req_b)]
        for fut in as_completed(futs):
            fut.result()
    conflicts = [
        item
        for item in outcomes
        if isinstance(item, KnownToolCapabilityRealizationConflictError)
    ]
    non_conflicts = [
        item
        for item in outcomes
        if isinstance(item, KnownToolCapabilityRealizationResult)
    ]
    assert len(conflicts) == 1
    assert len(non_conflicts) == 1
    assert activation.activate_calls <= 1
    winner = non_conflicts[0]
    if winner.outcome.name in {"REALIZED", "ALREADY_REALIZED"}:
        assert activation.activate_calls == 1


def test_concurrent_conflicting_host_one_binding() -> None:
    need = _need()
    domain, activation = _domain_with_resolver(_resolution())
    start = threading.Barrier(2)
    outcomes: list[object] = []

    def _run(request: KnownToolCapabilityRealizationRequest) -> None:
        start.wait(timeout=5)
        try:
            outcomes.append(domain.realize(request))
        except KnownToolCapabilityRealizationConflictError as exc:
            outcomes.append(exc)

    req_a = KnownToolCapabilityRealizationRequest(
        operation_id="op-conflict-host",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    req_b = req_a.model_copy(update={"host_profile_id": "host-profile-2"})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = [pool.submit(_run, req) for req in (req_a, req_b)]
        for fut in as_completed(futs):
            fut.result()
    conflicts = [
        item
        for item in outcomes
        if isinstance(item, KnownToolCapabilityRealizationConflictError)
    ]
    non_conflicts = [
        item
        for item in outcomes
        if isinstance(item, KnownToolCapabilityRealizationResult)
    ]
    assert len(conflicts) == 1
    assert len(non_conflicts) == 1
    assert activation.activate_calls <= 1
    winner = non_conflicts[0]
    if winner.outcome.name in {"REALIZED", "ALREADY_REALIZED"}:
        assert activation.activate_calls == 1


class _FlakyResolverForRetry:
    def __init__(self, resolution: ToolPackageResolution) -> None:
        self._resolution = resolution
        self.calls = 0

    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution:
        self.calls += 1
        if self.calls == 1:
            raise LookupError("first attempt failed")
        return self._resolution


def test_failed_attempt_then_concurrent_retries_single_in_flight() -> None:
    need = _need()
    resolver = _FlakyResolverForRetry(_resolution())
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    activation = _BlockingActivationGate(lifecycle)
    provider = Me14ToolCatalogProvider()
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    domain = ToolKnownCapabilityRealizationService(
        activation=activation,
        materializer=materializer,
        resolver=resolver,
    )
    request = KnownToolCapabilityRealizationRequest(
        operation_id="op-concurrent-retry",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    failed = domain.realize(request)
    assert failed.outcome.name == "FAILED"
    assert activation.activate_calls == 0
    results: list[KnownToolCapabilityRealizationResult] = []

    def _retry() -> None:
        results.append(domain.realize(request))

    threads = [threading.Thread(target=_retry) for _ in range(2)]
    for thread in threads:
        thread.start()
    activation.in_activate.wait(timeout=5)
    assert activation.activate_calls == 1
    activation.release_activate.set()
    for thread in threads:
        thread.join(timeout=5)
    assert activation.activate_calls == 1
    assert len(results) == 2
    assert all(r.outcome.name in {"REALIZED", "ALREADY_REALIZED"} for r in results)


def test_different_operation_ids_use_separate_operation_locks() -> None:
    need = _need()
    overlap = threading.Barrier(3)
    release = threading.Event()
    resolver_calls = 0
    resolver_calls_lock = threading.Lock()

    class _ParallelResolver:
        def resolve_for_identity(
            self,
            capability_identity: CapabilityIdentityKey,
        ) -> ToolPackageResolution:
            nonlocal resolver_calls
            with resolver_calls_lock:
                resolver_calls += 1
            overlap.wait(timeout=5)
            release.wait(timeout=5)
            return _resolution()

    class _NoRegistryActivation:
        host_profile_id = "host-profile-1"
        activate_calls = 0

        def registry_read(self):
            return ToolHostLifecycleService(host_profile_id="host-profile-1").registry

        def is_active(self, logical_tool_id: str) -> bool:
            return False

        def activation_metadata(self, logical_tool_id: str):
            return None

        def activate(self, **kwargs):
            self.activate_calls += 1
            return DomainLifecycleHandoffAck(
                disposition=DomainLifecycleHandoffDisposition.ACCEPTED,
                domain_reference="tool:parallel",
                reason_detail="accepted",
            )

    noop = _NoRegistryActivation()
    provider = Me14ToolCatalogProvider()
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-1")
    materializer = Me14ToolHostActivationMaterializer(
        lifecycle.registry,
        catalog_source_id=provider.catalog_source_id,
    )
    parallel_domain = ToolKnownCapabilityRealizationService(
        activation=noop,
        materializer=materializer,
        resolver=_ParallelResolver(),
    )
    req_a = KnownToolCapabilityRealizationRequest(
        operation_id="op-parallel-a",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    req_b = req_a.model_copy(update={"operation_id": "op-parallel-b"})

    def _run(req: KnownToolCapabilityRealizationRequest) -> None:
        parallel_domain.realize(req)

    threads = [threading.Thread(target=_run, args=(req,)) for req in (req_a, req_b)]
    for thread in threads:
        thread.start()
    overlap.wait(timeout=5)
    assert resolver_calls == 2
    release.set()
    for thread in threads:
        thread.join(timeout=5)
    assert noop.activate_calls == 2


class _ExceptionThenSuccessResolver:
    def __init__(self, resolution: ToolPackageResolution) -> None:
        self._resolution = resolution
        self.calls = 0

    def resolve_for_identity(
        self,
        capability_identity: CapabilityIdentityKey,
    ) -> ToolPackageResolution:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("resolver exploded")
        return self._resolution


def test_exception_in_resolver_releases_guard_and_allows_retry() -> None:
    need = _need()
    resolver = _ExceptionThenSuccessResolver(_resolution())
    domain, activation = _domain_with_custom_resolver(resolver)
    request = KnownToolCapabilityRealizationRequest(
        operation_id="op-exception-retry",
        host_profile_id="host-profile-1",
        capability_identity=need.capability_identity,
        requested_at=_CREATED,
    )
    with pytest.raises(RuntimeError):
        domain.realize(request)
    assert activation.activate_calls == 0
    second = domain.realize(request)
    assert second.outcome.name in {"REALIZED", "ALREADY_REALIZED"}
    assert activation.activate_calls == 1
    other_key = need.capability_identity.model_copy(update={"logical_id": "other.tool"})
    conflict = request.model_copy(update={"capability_identity": other_key})
    with pytest.raises(KnownToolCapabilityRealizationConflictError):
        domain.realize(conflict)
