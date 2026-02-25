# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from datetime import datetime, timezone

import pytest
from typing import Dict, Iterable, List

from intergrax.runtime.replay.replay_engine import ReplayEngine
from intergrax.runtime.replay.contracts.run_record_store import RunRecordStore
from intergrax.runtime.replay.contracts.trace_event_store import TraceEventStore
from intergrax.runtime.replay.contracts.artifact_store import ReplayArtifactStore
from intergrax.runtime.replay.contracts.run_record_dto import RunRecordDTO
from intergrax.runtime.replay.contracts.trace_event_dto import TraceEventDTO
from intergrax.runtime.replay.contracts.artifact_dto import ArtifactDTO


TENANT_A = "tenant_A"
TENANT_B = "tenant_B"
RUN_A = "run_A"
RUN_B = "run_B"


class FakeRunStore(RunRecordStore):
    def __init__(self, records: Dict[str, RunRecordDTO]) -> None:
        self._records = records

    def get(self, tenant_id: str, run_id: str) -> RunRecordDTO:
        key = (tenant_id, run_id)
        if key not in self._records:
            raise KeyError("Run not found")
        return self._records[key]


class FakeTraceStore(TraceEventStore):
    def __init__(self, traces: Dict[str, List[TraceEventDTO]]) -> None:
        self._traces = traces

    def get_events(self, tenant_id: str, run_id: str) -> Iterable[TraceEventDTO]:
        key = (tenant_id, run_id)
        return self._traces.get(key, [])


class FakeArtifactReader(ReplayArtifactStore):
    def __init__(self, artifacts: Dict[str, List[ArtifactDTO]]) -> None:
        self._artifacts = artifacts

    def list_for_run(self, tenant_id: str, run_id: str) -> Iterable[ArtifactDTO]:
        key = (tenant_id, run_id)
        return self._artifacts.get(key, [])


def test_replay_cross_tenant_isolation() -> None:
    # minimal DTOs
    
    NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

    run_record_a = RunRecordDTO(
        run_id=RUN_A,
        started_at=NOW,
        finished_at=NOW,
        status="finished",
        final_answer="ok",
    )

    run_record_b = RunRecordDTO(
        run_id=RUN_B,
        started_at=NOW,
        finished_at=NOW,
        status="finished",
        final_answer="ok",
    )

    run_store = FakeRunStore({
        (TENANT_A, RUN_A): run_record_a,
        (TENANT_B, RUN_B): run_record_b,
    })

    trace_store = FakeTraceStore({})
    artifact_reader = FakeArtifactReader({})

    engine = ReplayEngine(
        run_store=run_store,
        trace_store=trace_store,
        artifact_store=artifact_reader,
    )

    # Correct reconstruction
    engine.reconstruct(TENANT_A, RUN_A)
    engine.reconstruct(TENANT_B, RUN_B)

    # Cross-tenant attempt must fail
    with pytest.raises(KeyError):
        engine.reconstruct(TENANT_A, RUN_B)

    with pytest.raises(KeyError):
        engine.reconstruct(TENANT_B, RUN_A)