# © Artur Czarnecki. All rights reserved.

"""UCA-2 — capability realization contract validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
    derive_capability_realization_request_id,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)
from intergrax.contracts.capability_acquisition.evidence import (
    CapabilityRealizationEvidence,
)
from intergrax.contracts.capability_acquisition.outcome import (
    CapabilityRealizationOutcome,
)
from intergrax.contracts.capability_acquisition.reason_code import (
    CapabilityRealizationReasonCode,
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


def _tool_key(logical_id: str = "tools.alpha") -> CapabilityIdentityKey:
    return CapabilityIdentityKey(
        kind=CapabilityKind.TOOL,
        source_id="official.catalog",
        source_kind=CapabilitySourceKind.OFFICIAL,
        logical_id=logical_id,
    )


def _realization_need() -> CapabilityRealizationNeed:
    selected = _tool_key()
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        suitable_catalog_allowed_keys=(selected,),
        created_at=_CREATED,
    )
    return CapabilityRealizationNeed.from_discovery_completion(
        completion,
        capability_identity=selected,
    )


def test_request_id_is_deterministic() -> None:
    need = _realization_need()
    request_id = derive_capability_realization_request_id(
        realization_need_id=need.realization_need_id,
        request_nonce="nonce-1",
    )
    request = CapabilityRealizationRequest(
        request_id=request_id,
        request_nonce="nonce-1",
        realization_need=need,
        requested_at=_CREATED,
    )
    assert request.request_id == request_id


def test_succeeded_result_requires_availability_evidence() -> None:
    need = _realization_need()
    request_id = derive_capability_realization_request_id(
        realization_need_id=need.realization_need_id,
        request_nonce="nonce-1",
    )
    with pytest.raises(ValidationError):
        CapabilityRealizationResult(
            request_id=request_id,
            realization_need_id=need.realization_need_id,
            provider_id="provider.test",
            outcome=CapabilityRealizationOutcome.SUCCEEDED,
            reason_code=CapabilityRealizationReasonCode.NONE,
            capability_identity=need.capability_identity,
            started_at=_CREATED,
            completed_at=_CREATED,
        )


def test_succeeded_result_accepts_canonical_evidence() -> None:
    need = _realization_need()
    request_id = derive_capability_realization_request_id(
        realization_need_id=need.realization_need_id,
        request_nonce="nonce-1",
    )
    evidence = CapabilityRealizationEvidence.from_availability_evidence(
        CapabilityDiscoveryAvailabilityEvidence(
            host_available_keys=(need.capability_identity,),
        ),
        domain_reference="tool:tools.alpha@1.0.0:digest",
    )
    result = CapabilityRealizationResult(
        request_id=request_id,
        realization_need_id=need.realization_need_id,
        provider_id="provider.test",
        outcome=CapabilityRealizationOutcome.SUCCEEDED,
        reason_code=CapabilityRealizationReasonCode.NONE,
        capability_identity=need.capability_identity,
        started_at=_CREATED,
        completed_at=_CREATED,
        evidence=evidence,
    )
    assert result.outcome is CapabilityRealizationOutcome.SUCCEEDED
