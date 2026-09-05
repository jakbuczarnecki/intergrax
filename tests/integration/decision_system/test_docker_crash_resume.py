# © Artur Czarnecki. All rights reserved.

"""DS-E2E-06 — Docker/process crash + durable resume."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import pytest

from intergrax.contracts.decision_checkpoint import decision_checkpoint_state
from intergrax.contracts.decision_finalization import (
    decision_finalization_key,
    guard_decision_finalization,
    initial_decision_finalize_guard,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    AuthoritativeAcceptedDecision,
    DecisionArtifact,
    DecisionProposalRef,
    decision_lineage_ref,
    decision_version_lineage,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_revision import (
    decision_revision_checkpoint_state,
    decision_revision_policy,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.decision_checkpoint_persistence import save_decision_checkpoint
from intergrax.runtime.execution.decision_finalization_conformance import (
    IncidentDecisionPayload,
    conformance_artifact_payload_codec_registry,
)
from intergrax.runtime.execution.decision_recovery import (
    persist_terminal_decision_state,
    resume_decision_from_durable_state,
)
from intergrax.runtime.execution.sqlite_decision_checkpoint_persistence import (
    SQLiteDecisionCheckpointPersistence,
)
from intergrax.runtime.execution.sqlite_decision_finalization_persistence import (
    SQLiteDecisionFinalizationPersistence,
)
from intergrax.runtime.execution.decision_finalization_persistence import (
    DecisionDurableFinalizationDisposition,
)

from testing_support.decision_e2e.contracts import (
    DecisionE2EProofId,
    DecisionE2EQualificationResult,
    QualificationDisposition,
)
from testing_support.decision_e2e.evidence import lifecycle_stage_evidence

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.docker,
    pytest.mark.qualification,
    pytest.mark.no_ci,
    pytest.mark.slow,
]


def _identity(subject: str) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="decision_e2e", subject=subject),
        tenant_id="tenant-crash",
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _lifecycle_at_finalization(identity: DecisionIdentity):
    from intergrax.contracts.decision_lifecycle import (
        initial_decision_lifecycle_state,
        transition_decision_lifecycle,
    )

    state = initial_decision_lifecycle_state(identity)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.VERIFICATION)
    state = transition_decision_lifecycle(state, DecisionLifecycleStage.RESOLUTION)
    return transition_decision_lifecycle(state, DecisionLifecycleStage.FINALIZATION)


def _subprocess_crash_before_terminal(db_dir: str) -> None:
    identity = _identity("crash-before-terminal")
    proposal_ref = DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )
    revision = decision_revision_checkpoint_state(
        proposal_ref=proposal_ref,
        revision_count=1,
        max_revisions=2,
    )
    checkpoint = decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=initial_decision_finalize_guard(decision_finalization_key(identity)),
        revision=revision,
    )
    db_path = Path(db_dir)
    codecs = conformance_artifact_payload_codec_registry()
    save_decision_checkpoint(
        SQLiteDecisionCheckpointPersistence(
            db_path=db_path / "checkpoint.db",
            payload_codecs=codecs,
        ),
        checkpoint=checkpoint,
    )
    (db_path / "identity.json").write_text(
        json.dumps(
            {
                "tenant_id": identity.tenant_id,
                "decision_id": str(identity.decision_id),
                "namespace": identity.scope.namespace,
                "subject": identity.scope.subject,
            },
        ),
        encoding="utf-8",
    )


def _subprocess_resume(db_dir: str, queue: mp.Queue[tuple[str, int]]) -> None:
    db_path = Path(db_dir)
    sidecar = json.loads((db_path / "identity.json").read_text(encoding="utf-8"))
    from intergrax.contracts.decision_finalization import DecisionFinalizationKey
    from intergrax.contracts.decision_identity import validate_decision_id

    key = DecisionFinalizationKey(
        decision_id=validate_decision_id(sidecar["decision_id"]),
        scope=DecisionScope(namespace=sidecar["namespace"], subject=sidecar["subject"]),
        tenant_id=sidecar["tenant_id"],
    )
    codecs = conformance_artifact_payload_codec_registry()
    loaded = resume_decision_from_durable_state(
        checkpoint_persistence=SQLiteDecisionCheckpointPersistence(
            db_path=db_path / "checkpoint.db",
            payload_codecs=codecs,
        ),
        finalization_persistence=SQLiteDecisionFinalizationPersistence(
            db_path=db_path / "finalization.db",
            payload_codecs=codecs,
        ),
        key=key,
        runtime_revision_policy=decision_revision_policy(max_revisions=2),
    )
    stage = loaded.lifecycle.stage.value if loaded is not None else "missing"
    revision_count = loaded.revision.revision_count if loaded and loaded.revision else -1
    queue.put((stage, revision_count))


def _subprocess_authority_commit(db_dir: str) -> None:
    identity = _identity("authority-commit")
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="contain"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(identity.version)),
    )
    db_path = Path(db_dir)
    codecs = conformance_artifact_payload_codec_registry()
    finalization = SQLiteDecisionFinalizationPersistence(
        db_path=db_path / "finalization.db",
        payload_codecs=codecs,
    )
    commit = finalization.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=accepted,
    )
    assert commit.disposition is DecisionDurableFinalizationDisposition.COMMITTED
    checkpoint = decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard_decision_finalization(
            initial_decision_finalize_guard(decision_finalization_key(identity)),
            accepted,
        ).state,
    )
    save_decision_checkpoint(
        SQLiteDecisionCheckpointPersistence(
            db_path=db_path / "checkpoint.db",
            payload_codecs=codecs,
        ),
        checkpoint=checkpoint,
    )
    (db_path / "identity.json").write_text(
        json.dumps(
            {
                "tenant_id": identity.tenant_id,
                "decision_id": str(identity.decision_id),
                "namespace": identity.scope.namespace,
                "subject": identity.scope.subject,
            },
        ),
        encoding="utf-8",
    )


def test_ds_e2e_06_process_crash_resume(tmp_path: Path, decision_e2e_report_collector) -> None:
    db_dir = str(tmp_path)
    ctx = mp.get_context("spawn")

    writer = ctx.Process(target=_subprocess_crash_before_terminal, args=(db_dir,))
    writer.start()
    writer.join()
    assert writer.exitcode == 0

    result_queue: mp.Queue[tuple[str, int]] = ctx.Queue()
    reader = ctx.Process(target=_subprocess_resume, args=(db_dir, result_queue))
    reader.start()
    reader.join()
    assert reader.exitcode == 0
    stage, revision_count = result_queue.get(timeout=10)
    assert revision_count == 1

    authority_dir = str(tmp_path / "authority")
    Path(authority_dir).mkdir()
    auth_writer = ctx.Process(target=_subprocess_authority_commit, args=(authority_dir,))
    auth_writer.start()
    auth_writer.join()
    assert auth_writer.exitcode == 0

    sidecar = json.loads((Path(authority_dir) / "identity.json").read_text(encoding="utf-8"))
    from intergrax.contracts.decision_finalization import DecisionFinalizationKey
    from intergrax.contracts.decision_identity import validate_decision_id

    key = DecisionFinalizationKey(
        decision_id=validate_decision_id(sidecar["decision_id"]),
        scope=DecisionScope(
            namespace=sidecar["namespace"],
            subject=sidecar["subject"],
        ),
        tenant_id=sidecar["tenant_id"],
    )
    codecs = conformance_artifact_payload_codec_registry()
    checkpoint_store = SQLiteDecisionCheckpointPersistence(
        db_path=Path(authority_dir) / "checkpoint.db",
        payload_codecs=codecs,
    )
    finalization_store = SQLiteDecisionFinalizationPersistence(
        db_path=Path(authority_dir) / "finalization.db",
        payload_codecs=codecs,
    )
    loaded = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=key,
    )
    assert loaded is not None
    assert loaded.finalization.authoritative_outcome is not None
    terminal = persist_terminal_decision_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        checkpoint=loaded,
    )
    assert terminal.lifecycle.stage is DecisionLifecycleStage.TERMINAL

    decision_e2e_report_collector.record(
        DecisionE2EQualificationResult(
            proof_id=DecisionE2EProofId.DS_E2E_06,
            disposition=QualificationDisposition.PASSED,
            evidence=(lifecycle_stage_evidence(terminal.lifecycle.stage),),
            reason="sqlite durable subprocess crash/resume",
        ),
    )
