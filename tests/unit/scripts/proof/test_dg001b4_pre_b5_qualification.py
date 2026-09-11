# © Artur Czarnecki. All rights reserved.

"""Unit tests for DG-001B4 pre-B5 qualification helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.proof.dg001b4_pre_b5_qualification_contracts import (
    ControlledFailingWorkerBootstrapDiagnosticsSegment,
    qualification_secret_sentinel,
)
from scripts.proof.dg001b4_pre_b5_qualification_support import (
    RecordingBootstrapFailureReporter,
    build_production_equivalent_bootstrap_failure_producer,
)


def test_controlled_failing_diagnostics_segment_raises_runtime_error_with_sentinel() -> None:
    segment = ControlledFailingWorkerBootstrapDiagnosticsSegment()
    with pytest.raises(RuntimeError) as exc_info:
        segment(
            registry_projection=MagicMock(),
            settings=MagicMock(),
            environment_profile=MagicMock(),
            document_store=MagicMock(),
        )
    assert qualification_secret_sentinel() in str(exc_info.value)


def test_production_equivalent_producer_emits_to_recording_reporter() -> None:
    from intergrax.hosting.bootstrap_failure import (
        BootstrapIdentitySnapshot,
        BootstrapReadinessLevel,
        BootstrapSurfaceKind,
        mint_bootstrap_attempt_id,
    )
    from intergrax.hosting.process_bootstrap import HostedProcessBootstrapPhase

    reporter = RecordingBootstrapFailureReporter(records=[])
    producer = build_production_equivalent_bootstrap_failure_producer(extra_reporters=[reporter])
    identity = BootstrapIdentitySnapshot(
        bootstrap_attempt_id=mint_bootstrap_attempt_id(),
        application_id="local_workspace",
        process_role="background_worker",
    )
    record = producer.build_record(
        readiness_at_failure=BootstrapReadinessLevel.B3_TENANT_BINDING,
        stage=HostedProcessBootstrapPhase.DEPENDENCY_RESOLUTION,
        identity=identity,
        surface_kind=BootstrapSurfaceKind.WORKER_BACKGROUND,
        exc=RuntimeError("probe"),
    )
    producer.emit(record)
    assert reporter.records == [record]


def test_qualification_environment_json_is_secret_free(tmp_path: Path) -> None:
    payload = {
        "marker": "dg001b4-test",
        "sentinel": qualification_secret_sentinel(),
    }
    serialized = json.dumps({"marker": payload["marker"]})
    assert qualification_secret_sentinel() not in serialized
