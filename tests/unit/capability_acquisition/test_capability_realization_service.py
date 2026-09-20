# © Artur Czarnecki. All rights reserved.

"""UCA-2 — realization service dispatch and provider registry."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.capability_acquisition.service import CapabilityRealizationService
from intergrax.capability_acquisition.registry import CapabilityRealizationProviderRegistry
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
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)
from intergrax.contracts.capability_acquisition.errors import (
    CapabilityRealizationConfigurationError,
)
from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
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
from intergrax.contracts.capability_catalog.evidence import (
    CapabilityDiscoveryAvailabilityEvidence,
)

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 10, 0, tzinfo=UTC)


def _need(kind: CapabilityKind = CapabilityKind.TOOL) -> CapabilityRealizationNeed:
    key = CapabilityIdentityKey(
        kind=kind,
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id="cap.alpha",
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


def _request(
    need: CapabilityRealizationNeed,
    *,
    nonce: str = "nonce-1",
) -> CapabilityRealizationRequest:
    return CapabilityRealizationRequest(
        request_id=derive_capability_realization_request_id(
            realization_need_id=need.realization_need_id,
            request_nonce=nonce,
        ),
        request_nonce=nonce,
        realization_need=need,
        host_profile_id="host-1",
        requested_at=_CREATED,
    )


class _FakeProvider:
    def __init__(
        self,
        *,
        provider_id: str,
        kinds: frozenset[CapabilityKind],
        supports_override: bool | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._kinds = kinds
        self._supports_override = supports_override
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def supported_kinds(self) -> frozenset[CapabilityKind]:
        return self._kinds

    def supports(self, request: CapabilityRealizationRequest) -> bool:
        if self._supports_override is not None:
            return self._supports_override
        return request.capability_kind in self._kinds

    def realize(self, request: CapabilityRealizationRequest) -> CapabilityRealizationResult:
        self.calls += 1
        need = request.realization_need
        evidence = CapabilityRealizationEvidence.from_availability_evidence(
            CapabilityDiscoveryAvailabilityEvidence(
                host_available_keys=(need.capability_identity,),
            ),
        )
        return CapabilityRealizationResult(
            request_id=request.request_id,
            realization_need_id=need.realization_need_id,
            provider_id=self._provider_id,
            outcome=CapabilityRealizationOutcome.SUCCEEDED,
            reason_code=CapabilityRealizationReasonCode.NONE,
            capability_identity=need.capability_identity,
            started_at=_CREATED,
            completed_at=_CREATED,
            evidence=evidence,
        )


def test_no_provider_returns_not_supported() -> None:
    service = CapabilityRealizationService(())
    result = service.realize(_request(_need()))
    assert result.outcome is CapabilityRealizationOutcome.NOT_SUPPORTED
    assert result.reason_code is CapabilityRealizationReasonCode.NO_PROVIDER


def test_tool_provider_dispatched() -> None:
    provider = _FakeProvider(
        provider_id="fake.tool",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    service = CapabilityRealizationService((provider,))
    result = service.realize(_request(_need(CapabilityKind.TOOL)))
    assert result.outcome is CapabilityRealizationOutcome.SUCCEEDED
    assert result.provider_id == "fake.tool"
    assert provider.calls == 1


def test_skill_provider_dispatched() -> None:
    provider = _FakeProvider(
        provider_id="fake.skill",
        kinds=frozenset({CapabilityKind.SKILL}),
    )
    service = CapabilityRealizationService((provider,))
    result = service.realize(_request(_need(CapabilityKind.SKILL)))
    assert result.outcome is CapabilityRealizationOutcome.SUCCEEDED
    assert provider.calls == 1


def test_agent_provider_dispatched() -> None:
    provider = _FakeProvider(
        provider_id="fake.agent",
        kinds=frozenset({CapabilityKind.AGENT}),
    )
    service = CapabilityRealizationService((provider,))
    result = service.realize(_request(_need(CapabilityKind.AGENT)))
    assert result.outcome is CapabilityRealizationOutcome.SUCCEEDED
    assert provider.calls == 1


def test_duplicate_kind_registration_fails() -> None:
    first = _FakeProvider(provider_id="a", kinds=frozenset({CapabilityKind.TOOL}))
    second = _FakeProvider(provider_id="b", kinds=frozenset({CapabilityKind.TOOL}))
    with pytest.raises(CapabilityRealizationConfigurationError):
        CapabilityRealizationProviderRegistry((first, second))


def test_ambiguous_support_returns_conflict() -> None:
    tool = _FakeProvider(
        provider_id="tool",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    overlap = _FakeProvider(
        provider_id="overlap",
        kinds=frozenset({CapabilityKind.SKILL}),
        supports_override=True,
    )
    registry = CapabilityRealizationProviderRegistry((tool, overlap))
    eligible = registry.eligible_providers(_request(_need(CapabilityKind.TOOL)))
    assert len(eligible) == 2
    service = CapabilityRealizationService((tool, overlap))
    result = service.realize(_request(_need(CapabilityKind.TOOL)))
    assert result.outcome is CapabilityRealizationOutcome.CONFLICT


def test_idempotent_request_identity_stable() -> None:
    need = _need()
    first = _request(need, nonce="stable")
    second = _request(need, nonce="stable")
    assert first.request_id == second.request_id


def test_repeated_realize_same_request_no_duplicate_provider_side_effects() -> None:
    provider = _FakeProvider(
        provider_id="fake.tool",
        kinds=frozenset({CapabilityKind.TOOL}),
    )
    service = CapabilityRealizationService((provider,))
    request = _request(_need())
    service.realize(request)
    service.realize(request)
    assert provider.calls == 2
