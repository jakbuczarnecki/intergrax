# © Artur Czarnecki. All rights reserved.

"""UCA-4 — qualification service, registry, and policies."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from intergrax.capability_qualification.qualification_registry import (
    CapabilityQualificationProviderRegistry,
)
from intergrax.capability_qualification.qualification_service import (
    CapabilityQualificationService,
)
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
from intergrax.contracts.capability_qualification.errors import (
    CapabilityQualificationConfigurationError,
    CapabilityQualificationIntegrityError,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleOutcome,
    CapabilityQualificationLifecycleReasonCode,
)
from intergrax.contracts.capability_qualification.provider_descriptor import (
    CapabilityQualificationProviderDescriptor,
)
from intergrax.contracts.capability_qualification.provider_selection import (
    CapabilityQualificationProviderSelection,
    CapabilityQualificationProviderSelectionOutcome,
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


def _request(
    acquisition: CapabilityAcquisitionResult | None = None,
    *,
    nonce: str = "q-nonce-1",
) -> CapabilityQualificationRequest:
    gap = _gap()
    acq = acquisition or _succeeded_acquisition(gap)
    return CapabilityQualificationRequest(
        qualification_request_id=derive_capability_qualification_request_id(
            acquisition_request_id=acq.request_id,
            qualification_nonce=nonce,
        ),
        qualification_nonce=nonce,
        acquisition_request_id=acq.request_id,
        gap_id=acq.gap_id,
        strategy_id=acq.strategy_id or "strategy-1",
        acquisition_result=acq,
        correlation_id=acq.correlation_id,
        causation_id=acq.causation_id,
        requested_at=_CREATED,
    )


def _evidence(
    request: CapabilityQualificationRequest,
    provider_id: str,
) -> CapabilityQualificationEvidence:
    acq_evidence = request.acquisition_result.evidence
    return CapabilityQualificationEvidence(
        provider_id=provider_id,
        qualification_request_id=request.qualification_request_id,
        acquisition_request_id=request.acquisition_request_id,
        acquisition_strategy_id=request.strategy_id,
        gap_id=request.gap_id,
        artifact_reference=acq_evidence.artifact_reference if acq_evidence else None,
        domain_handoff_reference=(
            acq_evidence.domain_handoff_reference if acq_evidence else None
        ),
    )


class _FakeProvider:
    def __init__(
        self,
        *,
        provider_id: str,
        supports_override: bool | None = None,
    ) -> None:
        self._provider_id = provider_id
        self._supports_override = supports_override
        self.calls = 0

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def supports(self, request: CapabilityQualificationRequest) -> bool:
        if self._supports_override is not None:
            return self._supports_override
        return True

    def qualify(
        self, request: CapabilityQualificationRequest
    ) -> CapabilityQualificationResult:
        self.calls += 1
        return CapabilityQualificationResult(
            qualification_request_id=request.qualification_request_id,
            acquisition_request_id=request.acquisition_request_id,
            gap_id=request.gap_id,
            strategy_id=request.strategy_id,
            provider_id=self._provider_id,
            outcome=CapabilityQualificationOutcome.QUALIFIED,
            reason_code=CapabilityQualificationReasonCode.NONE,
            started_at=_CREATED,
            completed_at=_CREATED,
            evidence=_evidence(request, self._provider_id),
            correlation_id=request.correlation_id,
            causation_id=request.causation_id,
        )


def test_zero_providers_no_provider() -> None:
    service = CapabilityQualificationService(())
    decision = service.qualify(_request())
    assert (
        decision.qualification_result.outcome
        is CapabilityQualificationOutcome.NO_PROVIDER
    )


def test_single_provider_selected_and_called() -> None:
    provider = _FakeProvider(provider_id="fake-1")
    service = CapabilityQualificationService((provider,))
    decision = service.qualify(_request())
    assert (
        decision.qualification_result.outcome
        is CapabilityQualificationOutcome.QUALIFIED
    )
    assert provider.calls == 1
    assert decision.qualification_result.provider_id == "fake-1"


def test_multiple_providers_conflict() -> None:
    service = CapabilityQualificationService(
        (
            _FakeProvider(provider_id="a"),
            _FakeProvider(provider_id="b"),
        ),
    )
    decision = service.qualify(_request())
    assert (
        decision.qualification_result.outcome is CapabilityQualificationOutcome.CONFLICT
    )


def test_custom_provider_without_core_changes() -> None:
    provider = _FakeProvider(provider_id="tenant.custom")
    service = CapabilityQualificationService((provider,))
    decision = service.qualify(_request())
    assert decision.qualification_result.provider_id == "tenant.custom"


class _SelectBPolicy:
    def select(
        self,
        *,
        request: CapabilityQualificationRequest,
        candidates: tuple[CapabilityQualificationProviderDescriptor, ...],
    ) -> CapabilityQualificationProviderSelection:
        del request
        provider_b = next(c for c in candidates if c.provider_id == "provider-b")
        return CapabilityQualificationProviderSelection(
            outcome=CapabilityQualificationProviderSelectionOutcome.SELECTED,
            provider_id=provider_b.provider_id,
            reason_code=CapabilityQualificationReasonCode.NONE,
        )


def test_custom_selection_policy_picks_provider_b() -> None:
    service = CapabilityQualificationService(
        (
            _FakeProvider(provider_id="provider-a"),
            _FakeProvider(provider_id="provider-b"),
        ),
        selection_policy=_SelectBPolicy(),
    )
    decision = service.qualify(_request())
    assert decision.qualification_result.provider_id == "provider-b"


class _CustomLifecyclePolicy:
    def decide(self, *, request, qualification_result):
        from intergrax.contracts.capability_qualification.lifecycle_decision import (
            CapabilityQualificationLifecycleDecision,
        )

        del request, qualification_result
        return CapabilityQualificationLifecycleDecision(
            outcome=CapabilityQualificationLifecycleOutcome.QUARANTINE,
            reason_code=CapabilityQualificationLifecycleReasonCode.QUALIFICATION_BLOCKED,
            reason_detail="custom",
        )


def test_custom_lifecycle_policy() -> None:
    service = CapabilityQualificationService(
        (_FakeProvider(provider_id="p1"),),
        lifecycle_policy=_CustomLifecyclePolicy(),
    )
    decision = service.qualify(_request())
    assert (
        decision.lifecycle_decision.outcome
        is CapabilityQualificationLifecycleOutcome.QUARANTINE
    )


def test_subject_mismatch_raises_integrity_error() -> None:
    class _WrongSubjectProvider(_FakeProvider):
        def qualify(
            self, request: CapabilityQualificationRequest
        ) -> CapabilityQualificationResult:
            self.calls += 1
            bad_evidence = CapabilityQualificationEvidence(
                provider_id=self._provider_id,
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                acquisition_strategy_id=request.strategy_id,
                gap_id=request.gap_id,
                artifact_reference="artifact://not-acquired",
            )
            return CapabilityQualificationResult(
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                gap_id=request.gap_id,
                strategy_id=request.strategy_id,
                provider_id=self._provider_id,
                outcome=CapabilityQualificationOutcome.QUALIFIED,
                reason_code=CapabilityQualificationReasonCode.NONE,
                started_at=_CREATED,
                completed_at=_CREATED,
                evidence=bad_evidence,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )

    service = CapabilityQualificationService(
        (_WrongSubjectProvider(provider_id="bad"),)
    )
    with pytest.raises(CapabilityQualificationIntegrityError):
        service.qualify(_request())


def test_result_integrity_mismatch_raises() -> None:
    class _BadProvider(_FakeProvider):
        def qualify(
            self, request: CapabilityQualificationRequest
        ) -> CapabilityQualificationResult:
            self.calls += 1
            wrong_qreq_id = "capability-qualification-request:wrong:wrong"
            evidence = _evidence(request, self.provider_id).model_copy(
                update={"qualification_request_id": wrong_qreq_id},
            )
            return CapabilityQualificationResult(
                qualification_request_id=wrong_qreq_id,
                acquisition_request_id=request.acquisition_request_id,
                gap_id=request.gap_id,
                strategy_id=request.strategy_id,
                provider_id=self._provider_id,
                outcome=CapabilityQualificationOutcome.QUALIFIED,
                reason_code=CapabilityQualificationReasonCode.NONE,
                started_at=_CREATED,
                completed_at=_CREATED,
                evidence=evidence,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )

    service = CapabilityQualificationService((_BadProvider(provider_id="bad"),))
    with pytest.raises(CapabilityQualificationIntegrityError):
        service.qualify(_request())


def test_provider_qualified_without_evidence_raises_at_validation() -> None:
    class _NoEvidenceProvider(_FakeProvider):
        def qualify(
            self, request: CapabilityQualificationRequest
        ) -> CapabilityQualificationResult:
            self.calls += 1
            return CapabilityQualificationResult(
                qualification_request_id=request.qualification_request_id,
                acquisition_request_id=request.acquisition_request_id,
                gap_id=request.gap_id,
                strategy_id=request.strategy_id,
                provider_id=self._provider_id,
                outcome=CapabilityQualificationOutcome.QUALIFIED,
                reason_code=CapabilityQualificationReasonCode.NONE,
                started_at=_CREATED,
                completed_at=_CREATED,
                evidence=None,
                correlation_id=request.correlation_id,
                causation_id=request.causation_id,
            )

    service = CapabilityQualificationService((_NoEvidenceProvider(provider_id="x"),))
    decision = service.qualify(_request())
    assert (
        decision.qualification_result.outcome is CapabilityQualificationOutcome.FAILED
    )


def test_duplicate_provider_id_rejected() -> None:
    with pytest.raises(CapabilityQualificationConfigurationError):
        CapabilityQualificationProviderRegistry(
            (
                _FakeProvider(provider_id="dup"),
                _FakeProvider(provider_id="dup"),
            ),
        )


def test_provenance_audit_chain() -> None:
    req = _request()
    service = CapabilityQualificationService((_FakeProvider(provider_id="p1"),))
    decision = service.qualify(req)
    record = decision.audit_record
    assert record.acquisition_request_id == req.acquisition_request_id
    assert record.acquisition_strategy_id == req.strategy_id
    assert record.qualification_request_id == req.qualification_request_id
    assert record.qualification_provider_id == "p1"
    assert record.correlation_id == req.correlation_id
    assert record.causation_id == req.causation_id
    evidence = decision.qualification_result.evidence
    assert evidence is not None
    assert evidence.acquisition_request_id == req.acquisition_request_id
    assert evidence.acquisition_strategy_id == req.strategy_id


def test_failed_acquisition_cannot_build_request() -> None:
    gap = _gap()
    acq = _succeeded_acquisition(gap).model_copy(
        update={"outcome": CapabilityAcquisitionOutcome.FAILED},
    )
    with pytest.raises(ValidationError):
        _request(acq)


def test_qualified_lifecycle_accept_not_execution() -> None:
    service = CapabilityQualificationService((_FakeProvider(provider_id="p1"),))
    decision = service.qualify(_request())
    assert (
        decision.lifecycle_decision.outcome
        is CapabilityQualificationLifecycleOutcome.ACCEPT
    )
    assert (
        decision.qualification_result.outcome
        is CapabilityQualificationOutcome.QUALIFIED
    )
