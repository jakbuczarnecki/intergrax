# © Artur Czarnecki. All rights reserved.

"""AW-7C-P2/P3 scoped integration adaptation service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationError,
    ScopedIntegrationAdaptationFailureReason,
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationScope,
    ScopedIntegrationAdaptationTarget,
    ScopedIntegrationAdaptationTargetResolver,
    build_scoped_integration_adaptation_artifact,
    scoped_integration_adaptation_operation_id,
)
from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
    REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
    ReferenceScopedIntegrationAdaptationStrategy,
    reference_read_operation,
    reference_write_operation,
)
from intergrax.integrations.scoped_integration_adaptation_service import (
    ScopedIntegrationAdaptationPortAdapter,
    ScopedIntegrationAdaptationService,
)
from intergrax.integrations.scoped_integration_adaptation_target_resolver import (
    IntegrationIdentityScopedIntegrationAdaptationTargetResolver,
)

pytestmark = pytest.mark.unit

_TS = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_EXP = _TS + timedelta(hours=1)
_OP_READ = reference_read_operation()
_OP_WRITE = reference_write_operation()
_HOST_A = NetworkEgressHost(scheme="https", hostname="a.example.com", port=443)
_HOST_B = NetworkEgressHost(scheme="https", hostname="b.example.com", port=443)


class _Spec:
    specification_type = "demo"
    specification_version = "v1"
    specification_fingerprint = "sha256:spec"


def _scope(**kwargs: object) -> ScopedIntegrationAdaptationScope:
    defaults = {
        "tenant_id": "tenant-a",
        "integration_category": IntegrationCategory.MESSAGE_BUS,
        "provider_id": REFERENCE_SCOPED_INTEGRATION_ADAPTATION_PROVIDER_ID,
        "resource_scope": "rs-1",
        "permitted_operations": (_OP_READ, _OP_WRITE),
        "network_allowlist": NetworkEgressAllowlist(hosts=(_HOST_A, _HOST_B)),
        "credential_grant_ref": "grant-1",
        "expires_at": _EXP,
        "candidate_id": "cand-1",
        "candidate_revision": "rev-1",
    }
    defaults.update(kwargs)
    return ScopedIntegrationAdaptationScope(**defaults)  # type: ignore[arg-type]


def _request(scope: ScopedIntegrationAdaptationScope) -> ScopedIntegrationAdaptationRequest:
    return ScopedIntegrationAdaptationRequest(
        request_id="req-1",
        tenant_id=scope.tenant_id,
        integration_category=scope.integration_category,
        provider_id=scope.provider_id,
        resource_scope=scope.resource_scope,
        scope=scope,
    )


def _service(
    *,
    strategies: tuple[object, ...] | None = None,
    resolver: ScopedIntegrationAdaptationTargetResolver | None = None,
) -> ScopedIntegrationAdaptationService:
    selected = strategies or (ReferenceScopedIntegrationAdaptationStrategy(),)
    return ScopedIntegrationAdaptationService(
        target_resolver=resolver or IntegrationIdentityScopedIntegrationAdaptationTargetResolver(),
        strategies=selected,  # type: ignore[arg-type]
    )


def _artifact_for(scope: ScopedIntegrationAdaptationScope) -> ScopedIntegrationAdaptationArtifact:
    return build_scoped_integration_adaptation_artifact(
        artifact_id="art-1",
        tenant_id=scope.tenant_id,
        integration_category=scope.integration_category,
        provider_id=scope.provider_id,
        resource_scope=scope.resource_scope,
        strategy_id="strategy-1",
        candidate_id=scope.candidate_id,
        candidate_revision=scope.candidate_revision,
        scope=scope,
        specification=_Spec(),
    )


class _Strategy:
    def __init__(self, strategy_id: str, *, supports: bool = True) -> None:
        self._strategy_id = strategy_id
        self._supports = supports

    @property
    def strategy_id(self) -> str:
        return self._strategy_id

    def supports(
        self,
        request: ScopedIntegrationAdaptationRequest,
        target: ScopedIntegrationAdaptationTarget,
    ) -> bool:
        return self._supports

    def adapt(
        self,
        request: ScopedIntegrationAdaptationRequest,
        target: ScopedIntegrationAdaptationTarget,
    ) -> ScopedIntegrationAdaptationArtifact:
        return _artifact_for(request.scope)


class _FixedTargetResolver:
    def __init__(self, target: ScopedIntegrationAdaptationTarget) -> None:
        self._target = target

    def resolve_target(
        self,
        request: ScopedIntegrationAdaptationRequest,
    ) -> ScopedIntegrationAdaptationTarget:
        del request
        return self._target


def test_reference_strategy_narrows_scope() -> None:
    scope = _scope()
    artifact = _service().adapt(_request(scope))
    assert len(artifact.scope.permitted_operations) == 1
    assert len(artifact.scope.network_allowlist.hosts) == 1
    assert artifact.scope.expires_at < scope.expires_at


def test_single_strategy_selected() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    service = _service(strategies=(_Strategy("s1"),))
    artifact = service.adapt(_request(scope))
    assert artifact.artifact_id == "art-1"


def test_zero_strategies_unavailable() -> None:
    scope = _scope(provider_id="provider-1", permitted_operations=(_OP_READ,))
    service = _service(strategies=(_Strategy("s1", supports=False),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.STRATEGY_UNAVAILABLE


def test_ambiguous_strategies() -> None:
    scope = _scope(provider_id="provider-1", permitted_operations=(_OP_READ,))
    service = _service(strategies=(_Strategy("a"), _Strategy("b")))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.STRATEGY_AMBIGUOUS


def test_duplicate_strategy_id_rejected() -> None:
    with pytest.raises(ScopedIntegrationAdaptationError):
        _service(strategies=(_Strategy("dup"), _Strategy("dup")))


def test_replaceability_strategy_a_or_b() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    req = _request(scope)
    art_a = _service(strategies=(_Strategy("a"),)).adapt(req)
    art_b = _service(strategies=(_Strategy("b"),)).adapt(req)
    assert art_a.strategy_id == "strategy-1"
    assert art_b.strategy_id == "strategy-1"


def test_tenant_rewrite_rejected() -> None:
    scope = _scope()
    wide = _scope(tenant_id="tenant-b")

    class _BadStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(wide)

    service = _service(strategies=(_BadStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH


def test_target_resolver_mismatch_rejected() -> None:
    scope = _scope()
    bad_target = ScopedIntegrationAdaptationTarget(
        tenant_id="tenant-b",
        integration_category=scope.integration_category,
        provider_id=scope.provider_id,
        resource_scope=scope.resource_scope,
        current_revision=scope.candidate_revision,
    )
    service = _service(
        strategies=(_Strategy("s1"),),
        resolver=_FixedTargetResolver(bad_target),
    )
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH


def test_resolver_operational_failure_normalized() -> None:
    class _BrokenResolver:
        def resolve_target(
            self,
            request: ScopedIntegrationAdaptationRequest,
        ) -> ScopedIntegrationAdaptationTarget:
            del request
            raise OSError("resolver down")

    scope = _scope(provider_id="provider-1", permitted_operations=(_OP_READ,))
    service = _service(strategies=(_Strategy("s1"),), resolver=_BrokenResolver())
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.ADAPTATION_FAILED


def test_operation_widening_rejected() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    widened = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ, _OP_WRITE),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )

    class _WideStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(widened)

    service = _service(strategies=(_WideStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.OPERATION_WIDENING


def test_network_widening_rejected() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    widened = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A, _HOST_B)),
    )

    class _WideStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(widened)

    service = _service(strategies=(_WideStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.NETWORK_WIDENING


def test_extensible_operation_id_without_core_enum_change() -> None:
    custom = scoped_integration_adaptation_operation_id("CUSTOM_VENDOR_OPERATION")
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(custom,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    artifact = _service(strategies=(_Strategy("s1"),)).adapt(_request(scope))
    assert artifact.scope.permitted_operations[0].value == "CUSTOM_VENDOR_OPERATION"


def test_port_adapter_explicit_composition() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    port = ScopedIntegrationAdaptationPortAdapter(
        target_resolver=IntegrationIdentityScopedIntegrationAdaptationTargetResolver(),
        strategies=(_Strategy("s1"),),
    )
    assert port.adapt(_request(scope)).artifact_id == "art-1"


def test_reference_specification_fingerprint_mutates_with_semantics() -> None:
    from intergrax.integrations.qualification.reference_scoped_integration_adaptation import (
        ReferenceScopedIntegrationAdaptationSpecification,
    )

    first = ReferenceScopedIntegrationAdaptationSpecification(
        adaptation_intent="a",
        normalized_resource_scope="rs",
        operation_subset=("READ_CONFIGURATION",),
    )
    second = ReferenceScopedIntegrationAdaptationSpecification(
        adaptation_intent="b",
        normalized_resource_scope="rs",
        operation_subset=("READ_CONFIGURATION",),
    )
    assert first.specification_fingerprint != second.specification_fingerprint


def test_expiry_widening_rejected() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    widened = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
        expires_at=_EXP + timedelta(minutes=5),
    )

    class _WideStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(widened)

    service = _service(strategies=(_WideStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.EXPIRY_WIDENING


def test_revision_mismatch_rejected() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    bad = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
        candidate_revision="rev-2",
    )

    class _BadStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(bad)

    service = _service(strategies=(_BadStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.REVISION_MISMATCH


def test_credential_binding_substitution_rejected() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )
    bad = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
        credential_grant_ref="grant-2",
    )

    class _BadStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(bad)

    service = _service(strategies=(_BadStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.SCOPE_WIDENING


def test_empty_request_network_deny_scope_rejects_artifact_host() -> None:
    scope = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=()),
    )
    with_host = _scope(
        provider_id="provider-1",
        permitted_operations=(_OP_READ,),
        network_allowlist=NetworkEgressAllowlist(hosts=(_HOST_A,)),
    )

    class _BadStrategy(_Strategy):
        def adapt(
            self,
            request: ScopedIntegrationAdaptationRequest,
            target: ScopedIntegrationAdaptationTarget,
        ) -> ScopedIntegrationAdaptationArtifact:
            return _artifact_for(with_host)

    service = _service(strategies=(_BadStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.NETWORK_WIDENING


def test_reference_strategy_idempotent_fingerprint() -> None:
    scope = _scope()
    req = _request(scope)
    service = _service()
    first = service.adapt(req)
    second = service.adapt(req)
    assert first.artifact_fingerprint == second.artifact_fingerprint
