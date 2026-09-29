# © Artur Czarnecki. All rights reserved.

"""AW-7C-P2 scoped integration adaptation service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.sandbox_network_egress import (
    NetworkEgressAllowlist,
    NetworkEgressHost,
)
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationError,
    ScopedIntegrationAdaptationFailureReason,
    ScopedIntegrationAdaptationOperation,
    ScopedIntegrationAdaptationRequest,
    ScopedIntegrationAdaptationScope,
    ScopedIntegrationAdaptationTarget,
    build_scoped_integration_adaptation_artifact,
)
from intergrax.integrations.scoped_integration_adaptation_service import (
    ScopedIntegrationAdaptationService,
)

pytestmark = pytest.mark.unit

_TS = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)
_EXP = _TS + timedelta(hours=1)


class _Spec:
    specification_type = "demo"
    specification_version = "v1"
    specification_fingerprint = "sha256:spec"


def _scope(**kwargs: object) -> ScopedIntegrationAdaptationScope:
    defaults = {
        "tenant_id": "tenant-a",
        "integration_category": IntegrationCategory.MESSAGE_BUS,
        "provider_id": "provider-1",
        "resource_scope": "rs-1",
        "permitted_operations": (ScopedIntegrationAdaptationOperation.READ_CONFIGURATION,),
        "network_allowlist": NetworkEgressAllowlist(
            hosts=(
                NetworkEgressHost(
                    scheme="https",
                    hostname="api.example.com",
                    port=443,
                ),
            ),
        ),
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
        target=ScopedIntegrationAdaptationTarget(
            tenant_id=scope.tenant_id,
            integration_category=scope.integration_category,
            provider_id=scope.provider_id,
            resource_scope=scope.resource_scope,
            current_revision=scope.candidate_revision,
        ),
    )


def _artifact_for(scope: ScopedIntegrationAdaptationScope):
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


def test_single_strategy_selected() -> None:
    scope = _scope()
    service = ScopedIntegrationAdaptationService(strategies=(_Strategy("s1"),))
    artifact = service.adapt(_request(scope))
    assert artifact.artifact_id == "art-1"


def test_zero_strategies_unavailable() -> None:
    scope = _scope()
    service = ScopedIntegrationAdaptationService(strategies=(_Strategy("s1", supports=False),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.STRATEGY_UNAVAILABLE


def test_duplicate_strategy_id_rejected() -> None:
    with pytest.raises(ScopedIntegrationAdaptationError):
        ScopedIntegrationAdaptationService(
            strategies=(_Strategy("dup"), _Strategy("dup")),
        )


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

    service = ScopedIntegrationAdaptationService(strategies=(_BadStrategy("s1"),))
    with pytest.raises(ScopedIntegrationAdaptationError) as exc:
        service.adapt(_request(scope))
    assert exc.value.reason is ScopedIntegrationAdaptationFailureReason.TENANT_MISMATCH
