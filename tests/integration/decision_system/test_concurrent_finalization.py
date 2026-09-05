# © Artur Czarnecki. All rights reserved.

"""DS-E2E-07 — concurrent finalization race (multiprocess)."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from intergrax.contracts.decision_finalization import decision_finalization_key
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
    validate_decision_id,
)
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
)
from intergrax.runtime.execution.sqlite_decision_finalization_persistence import (
    SQLiteDecisionFinalizationPersistence,
)

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.qualification,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


def _worker(db_dir: str, mode: str, ready: mp.Barrier, result_queue: mp.Queue[str]) -> None:
    sidecar = json.loads((Path(db_dir) / "race.json").read_text(encoding="utf-8"))
    fixed_id = validate_decision_id(sidecar["decision_id"])
    scope = DecisionScope(namespace=sidecar["namespace"], subject=sidecar["subject"])
    tenant_id = sidecar["tenant_id"]
    execution = DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
    )
    identity_a = DecisionIdentity(
        decision_id=fixed_id,
        version=initial_decision_version(),
        scope=scope,
        tenant_id=tenant_id,
        execution=execution,
    )
    identity_b = DecisionIdentity(
        decision_id=fixed_id,
        version=next_decision_version(initial_decision_version()),
        scope=scope,
        tenant_id=tenant_id,
        execution=execution,
    )
    key = decision_finalization_key(identity_a)
    store = SQLiteDecisionFinalizationPersistence(
        db_path=Path(db_dir) / "finalization.db",
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    if mode == "winner":
        outcome = AuthoritativeAcceptedDecision(
            identity=identity_a,
            artifact=DecisionArtifact(
                kind=validate_decision_artifact_kind("incident_resolution"),
                content=IncidentDecisionPayload(recommendation="winner"),
            ),
            lineage=decision_version_lineage(
                current=decision_lineage_ref(identity_a.version),
            ),
        )
    else:
        outcome = AuthoritativeResolutionRecord(
            identity=identity_b,
            resolution=DecisionResolution.REJECTED,
        )
    ready.wait()
    disposition = store.commit_authoritative_outcome(
        key=key,
        requested_outcome=outcome,
    ).disposition
    result_queue.put(disposition.value)
    del store


def test_ds_e2e_07_concurrent_finalization_race(
    tmp_path: Path,
    decision_e2e_report_collector,
) -> None:
    db_dir = tmp_path / "race"
    db_dir.mkdir()
    fixed_id = mint_decision_id()
    (db_dir / "race.json").write_text(
        json.dumps(
            {
                "decision_id": str(fixed_id),
                "namespace": "decision_e2e",
                "subject": "race",
                "tenant_id": "tenant-race",
            },
        ),
        encoding="utf-8",
    )
    SQLiteDecisionFinalizationPersistence(
        db_path=db_dir / "finalization.db",
        payload_codecs=conformance_artifact_payload_codec_registry(),
    )
    ctx = mp.get_context("spawn")
    ready = ctx.Barrier(2, timeout=120)
    result_queue: mp.Queue[str] = ctx.Queue()
    workers = [
        ctx.Process(target=_worker, args=(str(db_dir), "winner", ready, result_queue)),
        ctx.Process(target=_worker, args=(str(db_dir), "loser", ready, result_queue)),
    ]
    try:
        for worker in workers:
            worker.start()
        dispositions: list[str] = []
        for _ in workers:
            dispositions.append(result_queue.get(timeout=120))
        for worker in workers:
            worker.join(timeout=30)
            assert worker.exitcode == 0, f"worker failed with exit code {worker.exitcode}"
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)

    assert DecisionDurableFinalizationDisposition.COMMITTED.value in dispositions
    conflict_or_idempotent = {
        DecisionDurableFinalizationDisposition.CONFLICT.value,
        DecisionDurableFinalizationDisposition.IDEMPOTENT_REPLAY.value,
    }
    assert any(item in conflict_or_idempotent for item in dispositions)

    codecs = conformance_artifact_payload_codec_registry()
    store = SQLiteDecisionFinalizationPersistence(
        db_path=db_dir / "finalization.db",
        payload_codecs=codecs,
    )
    key = decision_finalization_key(
        DecisionIdentity(
            decision_id=fixed_id,
            version=initial_decision_version(),
            scope=DecisionScope(namespace="decision_e2e", subject="race"),
            tenant_id="tenant-race",
            execution=DecisionExecutionLineage(
                task_id=mint_task_id(),
                run_id=mint_run_id(),
                attempt_id=mint_attempt_id(),
                execution_id=mint_execution_id(),
            ),
        ),
    )
    loaded = store.load_guard_state(key=key)
    assert loaded is not None
    assert loaded.authoritative_outcome is not None

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_07,
            disposition=QualificationDisposition.PASSED,
            evidence=(),
            reason=f"dispositions={dispositions}",
        ),
    )
