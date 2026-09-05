# © Artur Czarnecki. All rights reserved.

"""Qualification worker entrypoint for DS-E2E-06 Docker crash/resume."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

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


def _identity(subject: str) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="decision_e2e", subject=subject),
        tenant_id="tenant-docker-crash",
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


def _write_sidecar(db_path: Path, identity: DecisionIdentity) -> None:
    db_path.mkdir(parents=True, exist_ok=True)
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


def _load_sidecar(db_path: Path) -> dict[str, str]:
    return json.loads((db_path / "identity.json").read_text(encoding="utf-8"))


def _stores(db_path: Path):
    codecs = conformance_artifact_payload_codec_registry()
    checkpoint = SQLiteDecisionCheckpointPersistence(
        db_path=db_path / "checkpoint.db",
        payload_codecs=codecs,
    )
    finalization = SQLiteDecisionFinalizationPersistence(
        db_path=db_path / "finalization.db",
        payload_codecs=codecs,
    )
    return checkpoint, finalization, codecs


def _write_result(result_path: Path, payload: dict[str, object]) -> None:
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _wait_for_kill(*, signal_path: Path) -> None:
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.write_text(json.dumps({"ready": True, "pid": None}), encoding="utf-8")
    while True:
        time.sleep(1.0)


def checkpoint_persist(db_dir: Path, signal_path: Path) -> None:
    identity = _identity("docker-checkpoint")
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
    checkpoint_store, _, _ = _stores(db_dir)
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    _write_sidecar(db_dir, identity)
    _wait_for_kill(signal_path=signal_path)


def checkpoint_resume(db_dir: Path, result_path: Path) -> None:
    from intergrax.contracts.decision_finalization import DecisionFinalizationKey
    from intergrax.contracts.decision_identity import validate_decision_id

    sidecar = _load_sidecar(db_dir)
    key = DecisionFinalizationKey(
        decision_id=validate_decision_id(sidecar["decision_id"]),
        scope=DecisionScope(namespace=sidecar["namespace"], subject=sidecar["subject"]),
        tenant_id=sidecar["tenant_id"],
    )
    checkpoint_store, finalization_store, _ = _stores(db_dir)
    loaded = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=key,
        runtime_revision_policy=decision_revision_policy(max_revisions=2),
    )
    stage = loaded.lifecycle.stage.value if loaded is not None else "missing"
    revision_count = loaded.revision.revision_count if loaded and loaded.revision else -1
    _write_result(
        result_path,
        {
            "window": "checkpoint",
            "stage": stage,
            "revision_count": revision_count,
        },
    )


def authority_commit(db_dir: Path, signal_path: Path) -> None:
    identity = _identity("docker-authority")
    accepted = AuthoritativeAcceptedDecision(
        identity=identity,
        artifact=DecisionArtifact(
            kind=validate_decision_artifact_kind("incident_resolution"),
            content=IncidentDecisionPayload(recommendation="contain"),
        ),
        lineage=decision_version_lineage(current=decision_lineage_ref(identity.version)),
    )
    checkpoint_store, finalization_store, _ = _stores(db_dir)
    commit = finalization_store.commit_authoritative_outcome(
        key=decision_finalization_key(identity),
        requested_outcome=accepted,
    )
    if commit.disposition is not DecisionDurableFinalizationDisposition.COMMITTED:
        raise RuntimeError(f"authority commit failed: {commit.disposition.value}")
    checkpoint = decision_checkpoint_state(
        lifecycle=_lifecycle_at_finalization(identity),
        finalization=guard_decision_finalization(
            initial_decision_finalize_guard(decision_finalization_key(identity)),
            accepted,
        ).state,
    )
    save_decision_checkpoint(checkpoint_store, checkpoint=checkpoint)
    _write_sidecar(db_dir, identity)
    _wait_for_kill(signal_path=signal_path)


def authority_resume(db_dir: Path, result_path: Path) -> None:
    from intergrax.contracts.decision_finalization import DecisionFinalizationKey
    from intergrax.contracts.decision_identity import validate_decision_id

    sidecar = _load_sidecar(db_dir)
    key = DecisionFinalizationKey(
        decision_id=validate_decision_id(sidecar["decision_id"]),
        scope=DecisionScope(namespace=sidecar["namespace"], subject=sidecar["subject"]),
        tenant_id=sidecar["tenant_id"],
    )
    checkpoint_store, finalization_store, _ = _stores(db_dir)
    loaded = resume_decision_from_durable_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        key=key,
    )
    if loaded is None:
        raise RuntimeError("resume returned no durable checkpoint")
    if loaded.finalization.authoritative_outcome is None:
        raise RuntimeError("authoritative outcome missing after resume")
    authority_id = str(loaded.finalization.authoritative_outcome.identity.decision_id)
    terminal = persist_terminal_decision_state(
        checkpoint_persistence=checkpoint_store,
        finalization_persistence=finalization_store,
        checkpoint=loaded,
    )
    _write_result(
        result_path,
        {
            "window": "authority",
            "stage": terminal.lifecycle.stage.value,
            "authority_decision_id": authority_id,
            "duplicate_authority": False,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DS-E2E Docker qualification worker")
    parser.add_argument(
        "phase",
        choices=(
            "checkpoint-persist",
            "checkpoint-resume",
            "authority-commit",
            "authority-resume",
        ),
    )
    parser.add_argument("--db-dir", required=True)
    parser.add_argument("--signal", default="")
    parser.add_argument("--result", default="")
    args = parser.parse_args(argv)
    db_dir = Path(args.db_dir)
    signal_path = Path(args.signal) if args.signal else db_dir / "ready.json"
    result_path = Path(args.result) if args.result else db_dir / "result.json"

    if args.phase == "checkpoint-persist":
        checkpoint_persist(db_dir, signal_path)
    elif args.phase == "checkpoint-resume":
        checkpoint_resume(db_dir, result_path)
    elif args.phase == "authority-commit":
        authority_commit(db_dir, signal_path)
    elif args.phase == "authority-resume":
        authority_resume(db_dir, result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
