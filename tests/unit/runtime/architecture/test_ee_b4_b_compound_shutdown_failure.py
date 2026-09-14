# © Artur Czarnecki. All rights reserved.

"""EE-B4-B — compound shutdown failures preserve primary authority."""

from __future__ import annotations

import pytest

from testing_support.chaos.barriers import PhaseGate
from testing_support.shutdown.models import (
    ReferenceRootAdmissionDecision,
    ReferenceShutdownFailureKind,
)
from testing_support.shutdown.ports import (
    RecordingMandatoryEvidenceFlush,
    RecordingObservabilityExporter,
)
from tests.unit.runtime.architecture._ee_b4_b_lifecycle import make_lifecycle

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_ee_b4_b_worker_primary_observability_secondary() -> None:
    lifecycle = make_lifecycle()
    lifecycle.observability_exporter = RecordingObservabilityExporter(fail_on_call=1)
    gate = PhaseGate()
    decision, permit = await lifecycle.try_admit_root()
    assert decision is ReferenceRootAdmissionDecision.ADMITTED
    assert permit is not None

    async def fail_work() -> None:
        gate.mark_started("w")
        raise RuntimeError("worker")

    handle = await lifecycle.run_root_work(
        execution_id="compound",
        permit=permit,
        work_factory=fail_work,
    )
    await gate.wait_until_started(frozenset({"w"}))
    outcome = await lifecycle.shutdown()
    with pytest.raises(RuntimeError, match="worker"):
        await handle.worker_task
    assert outcome.primary_failure_kind is ReferenceShutdownFailureKind.WORKER
    assert (
        ReferenceShutdownFailureKind.OBSERVABILITY_EXPORT
        in outcome.secondary_failure_kinds
    )


@pytest.mark.asyncio
async def test_ee_b4_b_worker_primary_mandatory_evidence_secondary() -> None:
    lifecycle = make_lifecycle()
    lifecycle.mandatory_evidence = RecordingMandatoryEvidenceFlush(fail_on_call=1)
    gate = PhaseGate()
    decision, permit = await lifecycle.try_admit_root()
    assert permit is not None

    async def fail_work() -> None:
        gate.mark_started("w")
        raise RuntimeError("worker")

    handle = await lifecycle.run_root_work(
        execution_id="compound-ev",
        permit=permit,
        work_factory=fail_work,
    )
    await gate.wait_until_started(frozenset({"w"}))
    outcome = await lifecycle.shutdown()
    with pytest.raises(RuntimeError, match="worker"):
        await handle.worker_task
    assert outcome.primary_failure_kind is ReferenceShutdownFailureKind.WORKER
    assert (
        ReferenceShutdownFailureKind.MANDATORY_EVIDENCE
        in outcome.secondary_failure_kinds
    )
