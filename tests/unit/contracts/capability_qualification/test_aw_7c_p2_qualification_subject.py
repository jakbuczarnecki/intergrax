# © Artur Czarnecki. All rights reserved.

"""AW-7C-P2 capability qualification subject contracts."""

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
from intergrax.contracts.capability_acquisition.acquisition_result import (
    CapabilityAcquisitionResult,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    build_acquisition_qualification_request,
    derive_capability_qualification_request_id,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubjectKind,
    derive_acquisition_subject_integrity_fingerprint,
    project_acquisition_qualification_subject,
)

pytestmark = pytest.mark.unit

_TS = datetime(2026, 9, 20, 12, 0, tzinfo=UTC)


def _acquisition() -> CapabilityAcquisitionResult:
    return CapabilityAcquisitionResult(
        request_id="acq-1",
        gap_id="gap-1",
        strategy_id="strategy-1",
        outcome=CapabilityAcquisitionOutcome.SUCCEEDED,
        reason_code=CapabilityAcquisitionReasonCode.NONE,
        started_at=_TS,
        completed_at=_TS,
        evidence=CapabilityAcquisitionEvidence(artifact_reference="artifact://a"),
        correlation_id="corr-1",
        causation_id="cause-1",
    )


def test_acquisition_projection_deterministic() -> None:
    acq = _acquisition()
    left = project_acquisition_qualification_subject(acquisition_result=acq)
    right = project_acquisition_qualification_subject(acquisition_result=acq)
    assert left == right
    assert left.subject_kind is CapabilityQualificationSubjectKind.ACQUIRED_CAPABILITY
    assert left.subject_id == "acq-1"


def test_request_id_stable_for_acquisition_subject() -> None:
    acq = _acquisition()
    request = build_acquisition_qualification_request(
        acquisition_result=acq,
        qualification_nonce="qual-1",
        requested_at=_TS,
    )
    expected = derive_capability_qualification_request_id(
        subject_id="acq-1",
        qualification_nonce="qual-1",
    )
    assert request.qualification_request_id == expected


def test_fingerprint_changes_when_tenant_would_apply_to_adaptation_only() -> None:
    fp1 = derive_acquisition_subject_integrity_fingerprint(
        acquisition_request_id="acq-1",
        gap_id="gap-1",
        strategy_id="s1",
        artifact_reference="a",
        domain_handoff_reference=None,
        correlation_id=None,
        causation_id=None,
    )
    fp2 = derive_acquisition_subject_integrity_fingerprint(
        acquisition_request_id="acq-2",
        gap_id="gap-1",
        strategy_id="s1",
        artifact_reference="a",
        domain_handoff_reference=None,
        correlation_id=None,
        causation_id=None,
    )
    assert fp1 != fp2
