# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — mandatory evidence flush fail-closed semantics."""

from __future__ import annotations

import pytest

from testing_support.shutdown.models import ReferenceShutdownFailureKind
from testing_support.shutdown.ports import RecordingMandatoryEvidenceFlush
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_mandatory_evidence_flush_success_allows_clean_shutdown() -> None:
    lifecycle = make_lifecycle()
    outcome = await lifecycle.shutdown()
    assert lifecycle.mandatory_evidence.flushed
    assert outcome.clean_success


@pytest.mark.asyncio
async def test_ee_b4_b_mandatory_evidence_failure_not_clean_success() -> None:
    lifecycle = make_lifecycle()
    lifecycle.mandatory_evidence = RecordingMandatoryEvidenceFlush(fail_on_call=1)
    outcome = await lifecycle.shutdown()
    assert (
        outcome.primary_failure_kind is ReferenceShutdownFailureKind.MANDATORY_EVIDENCE
    )
    assert not outcome.clean_success
