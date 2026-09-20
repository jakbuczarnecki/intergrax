# © Artur Czarnecki. All rights reserved.

"""UCA-4 — qualification request/result contract invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.capability_acquisition.acquisition_evidence import (
    CapabilityAcquisitionEvidence,
)
from intergrax.contracts.capability_acquisition.acquisition_outcome import (
    CapabilityAcquisitionOutcome,
)
from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    derive_capability_acquisition_request_id,
)
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_catalog.capability_gap import CapabilityGap
from intergrax.contracts.capability_catalog.discovery_completion import (
    build_discovery_completion,
)
from intergrax.contracts.capability_catalog.federation import (
    CapabilityCatalogFederationCompleteness,
)
from intergrax.contracts.capability_qualification.qualification_evidence import (
    CapabilityQualificationEvidence,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.capability_qualification.qualification_result import (
    CapabilityQualificationResult,
)

pytestmark = pytest.mark.unit

_CREATED = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _gap() -> CapabilityGap:
    completion = build_discovery_completion(
        need_id="need-1",
        discovery_correlation_id="corr-1",
        federation_completeness=CapabilityCatalogFederationCompleteness.COMPLETE,
        created_at=_CREATED,
    )
    return CapabilityGap.from_discovery_completion(completion)


def _succeeded_acquisition(gap: CapabilityGap) -> CapabilityAcquisitionResult:
    acq_id = derive_capability_acquisition_request_id(
        gap_id=gap.gap_id,
        request_nonce="nonce-1",
    )
    return CapabilityAcquisitionResult(
        request_id=acq_id,
        gap_id=gap.gap_id,
        strategy_id="strategy-1",
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_CREATED,
        completed_at=_CREATED,
        evidence=CapabilityAcquisitionEvidence(artifact_reference="artifact://a"),
        correlation_id="corr-acq",
        causation_id="cause-acq",
    )


def _qualification_request(
    acquisition: CapabilityAcquisitionResult,
    *,
    nonce: str = "q-nonce-1",
) -> CapabilityQualificationRequest:
    return CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=acquisition.request_id,
            qualification_nonce=nonce,
        ),
        qualification_nonce=nonce,
        acquisition_request_id=acquisition.request_id,
        gap_id=acquisition.gap_id,
        strategy_id=acquisition.strategy_id or "strategy-1",
        acquisition_result=acquisition,
        correlation_id=acquisition.correlation_id,
        causation_id=acquisition.causation_id,
        requested_at=_CREATED,
    )


def test_succeeded_acquisition_accepted_for_qualification() -> None:
    gap = _gap()
    request = _qualification_request(_succeeded_acquisition(gap))
    assert request.acquisition_result.outcome is CapabilityAcquisitionOutcome.SUCCEEDED


def test_failed_acquisition_rejected_at_request_boundary() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    failed = acquisition.model_copy(
        update={"outcome": CapabilityAcquisitionOutcome.FAILED},
    )
    with pytest.raises(ValueError, match="SUCCEEDED"):
        _qualification_request(failed)


def test_qualified_requires_evidence() -> None:
    with pytest.raises(ValueError, match="qualification evidence"):
        CapabilityQualificationResult(
            qualification_request_id="capability-qualification-request:a:b",
            acquisition_request_id="capability-acquisition-request:g:n",
            gap_id="capability-gap:need-1:corr-1",
            strategy_id="strategy-1",
            provider_id="provider-1",
            outcome=CapabilityQualificationOutcome.QUALIFIED,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=_CREATED,
            completed_at=_CREATED,
        )


def test_qualified_with_evidence_ok() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap)
    qreq_id = derive_capability_qualification_request_id(
        acquisition_request_id=acq.request_id,
        qualification_nonce="n",
    )
    evidence = CapabilityQualificationEvidence(
        provider_id="provider-1",
        qualification_request_id=qreq_id,
        acquisition_request_id=acq.request_id,
        acquisition_strategy_id="strategy-1",
        gap_id=gap.gap_id,
        artifact_reference="artifact://a",
    )
    CapabilityQualificationResult(
        qualification_request_id=qreq_id,
        acquisition_request_id=acq.request_id,
        gap_id=gap.gap_id,
        strategy_id="strategy-1",
        provider_id="provider-1",
        outcome=CapabilityQualificationOutcome.QUALIFIED,
        reason_code=CapabilityQualificationReasonCode.NONE,
        started_at=_CREATED,
        completed_at=_CREATED,
        evidence=evidence,
        correlation_id="corr-acq",
        causation_id="cause-acq",
    )


def test_request_rejects_mismatched_gap_id() -> None:
    gap = _gap()
    acquisition = _succeeded_acquisition(gap)
    with pytest.raises(ValueError, match="gap_id"):
        CapabilityQualificationRequest(
            qualification_request_id=derive_capability_qualification_request_id(
                acquisition_request_id=acquisition.request_id,
                qualification_nonce="n",
            ),
            qualification_nonce="n",
            acquisition_request_id=acquisition.request_id,
            gap_id="wrong-gap",
            strategy_id="strategy-1",
            acquisition_result=acquisition,
            requested_at=_CREATED,
        )
