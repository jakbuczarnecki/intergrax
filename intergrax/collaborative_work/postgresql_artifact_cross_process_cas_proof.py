# © Artur Czarnecki. All rights reserved.

"""Cross-process PostgreSQL artifact publication CAS proof helpers for Collaborative Work MP-3E."""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.collaborative_work.persistence import open_postgresql_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    PublishWorkArtifactVersionCommand,
    WorkArtifactRevisionConflict,
)
from intergrax.contracts.collaborative_work import ArtifactContentRef
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)

_WORKER_JOIN_TIMEOUT_SECONDS = 30.0
_BARRIER_TIMEOUT_SECONDS = 10.0
_RESULT_TIMEOUT_SECONDS = 30.0


class CrossProcessArtifactPublicationProofFailure(Exception):
    """Cross-process artifact publication proof rejected by semantic invariants."""


@dataclass(frozen=True, slots=True)
class CrossProcessArtifactPublicationProofResult:
    """Observed cross-process artifact publication race outcome."""

    worker_pids: tuple[int, ...]
    successes: int
    conflicts: int
    winning_version_id: str
    losing_version_id: str
    final_revision: int
    final_current_version_id: str
    history_version_ids: tuple[str, ...]


def _cross_process_artifact_publication_worker(
    dsn: str,
    schema_name: str,
    tenant_id: str,
    workspace_id: str,
    work_item_id: str,
    work_artifact_id: str,
    work_artifact_version_id: str,
    expected_revision: int,
    idempotency_key: str,
    content_ref_json: str,
    created_at_iso: str,
    published_at_iso: str,
    artifact_updated_at_iso: str,
    barrier: mp.synchronize.Barrier,
    result_queue: mp.Queue[tuple[str, ...]],
) -> None:
    bundle = None
    try:
        config = PostgreSQLIntegrationConfig(dsn=dsn)
        bundle = open_postgresql_collaborative_work_repositories(
            config=config,
            schema_name=schema_name,
        )
        content_ref = ArtifactContentRef.model_validate_json(content_ref_json)
        barrier.wait(timeout=_BARRIER_TIMEOUT_SECONDS)
        bundle.publication.publish_version(
            PublishWorkArtifactVersionCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                work_item_id=work_item_id,
                work_artifact_id=work_artifact_id,
                work_artifact_version_id=work_artifact_version_id,
                expected_revision=expected_revision,
                created_by_principal_id="principal-creator",
                published_by_principal_id="principal-publisher",
                content_ref=content_ref,
                created_at=datetime.fromisoformat(created_at_iso),
                published_at=datetime.fromisoformat(published_at_iso),
                artifact_updated_at=datetime.fromisoformat(artifact_updated_at_iso),
                idempotency_key=idempotency_key,
            ),
        )
        result_queue.put(("published", work_artifact_version_id, str(os.getpid())))
    except WorkArtifactRevisionConflict:
        result_queue.put(("conflict", work_artifact_version_id, str(os.getpid())))
    except BaseException as exc:  # noqa: BLE001
        result_queue.put(("error", type(exc).__name__, str(exc), str(os.getpid())))
    finally:
        if bundle is not None:
            bundle.close()


def run_postgresql_artifact_cross_process_cas_proof(
    *,
    config: PostgreSQLIntegrationConfig,
    schema_name: str,
    tenant_id: str,
    workspace_id: str,
    work_item_id: str,
    work_artifact_id: str,
    initial_version_id: str,
    winning_version_id: str,
    losing_version_id: str,
    expected_revision: int,
    published_at: datetime,
    artifact_updated_at: datetime,
    content_ref: ArtifactContentRef,
) -> CrossProcessArtifactPublicationProofResult:
    """
    Prove PostgreSQL artifact publication CAS across two distinct OS processes.

    Each worker independently materializes repositories and races on
    ``publish_version(expected_revision=N)`` with distinct version identities.
    """
    dsn = config.connection_string()
    if not dsn:
        raise CrossProcessArtifactPublicationProofFailure(
            "PostgreSQL connection string is required",
        )

    bundle_a = open_postgresql_collaborative_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    bundle_b = open_postgresql_collaborative_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        read_a = bundle_a.artifact.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_artifact_id=work_artifact_id,
        )
        read_b = bundle_b.artifact.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_artifact_id=work_artifact_id,
        )
        if read_a is None or read_b is None:
            raise CrossProcessArtifactPublicationProofFailure(
                "cross-process pre-read missing work artifact",
            )
        if not (
            read_a.revision
            == read_b.revision
            == expected_revision
        ):
            raise CrossProcessArtifactPublicationProofFailure(
                "cross-process revision baseline mismatch",
            )
        if read_a.current_version_id.strip() != initial_version_id.strip():
            raise CrossProcessArtifactPublicationProofFailure(
                "cross-process initial pointer mismatch",
            )
    finally:
        bundle_a.close()
        bundle_b.close()

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2, timeout=_BARRIER_TIMEOUT_SECONDS)
    result_queue: mp.Queue[tuple[str, ...]] = ctx.Queue()
    content_ref_json = content_ref.model_dump_json()
    published_at_iso = published_at.astimezone(UTC).isoformat()
    artifact_updated_at_iso = artifact_updated_at.astimezone(UTC).isoformat()
    created_at_iso = published_at_iso
    worker_specs = (
        (winning_version_id, "artifact-cross-process-win"),
        (losing_version_id, "artifact-cross-process-lose"),
    )
    processes = [
        ctx.Process(
            target=_cross_process_artifact_publication_worker,
            args=(
                dsn,
                schema_name,
                tenant_id,
                workspace_id,
                work_item_id,
                work_artifact_id,
                version_id,
                expected_revision,
                idempotency_key,
                content_ref_json,
                created_at_iso,
                published_at_iso,
                artifact_updated_at_iso,
                barrier,
                result_queue,
            ),
        )
        for version_id, idempotency_key in worker_specs
    ]
    outcomes: list[tuple[str, ...]] = []
    try:
        for process in processes:
            process.start()
        for _ in processes:
            outcomes.append(result_queue.get(timeout=_RESULT_TIMEOUT_SECONDS))
        for process in processes:
            process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
            if process.exitcode != 0:
                raise CrossProcessArtifactPublicationProofFailure(
                    f"cross-process worker exited with code {process.exitcode}",
                )
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()

    if len(outcomes) != 2:
        raise CrossProcessArtifactPublicationProofFailure("expected exactly two worker outcomes")

    worker_pids: list[int] = []
    successes = 0
    conflicts = 0
    published_version_ids: list[str] = []
    for outcome in outcomes:
        status = outcome[0]
        if status == "published":
            successes += 1
            published_version_ids.append(outcome[1])
            worker_pids.append(int(outcome[2]))
            continue
        if status == "conflict":
            conflicts += 1
            worker_pids.append(int(outcome[2]))
            continue
        raise CrossProcessArtifactPublicationProofFailure(
            f"unexpected worker outcome: {outcome}",
        )

    if successes != 1 or conflicts != 1:
        raise CrossProcessArtifactPublicationProofFailure(
            f"expected one success and one conflict, got successes={successes} conflicts={conflicts}",
        )
    if len(set(worker_pids)) != 2:
        raise CrossProcessArtifactPublicationProofFailure("expected distinct worker PIDs")

    winning_version_id_observed = published_version_ids[0]
    losing_version_id_observed = (
        winning_version_id
        if losing_version_id == winning_version_id_observed
        else losing_version_id
    )
    if winning_version_id_observed not in {winning_version_id, losing_version_id}:
        raise CrossProcessArtifactPublicationProofFailure("unexpected winning version id")

    verify_bundle = open_postgresql_collaborative_work_repositories(
        config=config,
        schema_name=schema_name,
    )
    try:
        final_artifact = verify_bundle.artifact.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_artifact_id=work_artifact_id,
        )
        if final_artifact is None:
            raise CrossProcessArtifactPublicationProofFailure(
                "final work artifact missing after cross-process race",
            )
        if final_artifact.revision != expected_revision + 1:
            raise CrossProcessArtifactPublicationProofFailure(
                "cross-process final revision mismatch",
            )
        if final_artifact.current_version_id.strip() != winning_version_id_observed.strip():
            raise CrossProcessArtifactPublicationProofFailure(
                "cross-process final pointer mismatch",
            )

        winning_version = verify_bundle.version.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_artifact_version_id=winning_version_id_observed,
        )
        if winning_version is None:
            raise CrossProcessArtifactPublicationProofFailure("winning version missing")

        orphan_loser_id = (
            losing_version_id
            if winning_version_id_observed == winning_version_id
            else winning_version_id
        )
        orphan = verify_bundle.version.get(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_artifact_version_id=orphan_loser_id,
        )
        if orphan is not None:
            raise CrossProcessArtifactPublicationProofFailure("losing version orphan present")

        history = verify_bundle.version.list_for_artifact(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            work_artifact_id=work_artifact_id,
        )
        history_version_ids = tuple(
            record.work_artifact_version_id.strip() for record in history
        )
        expected_history = (initial_version_id.strip(), winning_version_id_observed.strip())
        if history_version_ids != expected_history:
            raise CrossProcessArtifactPublicationProofFailure(
                f"cross-process history mismatch: {history_version_ids}",
            )
    finally:
        verify_bundle.close()

    return CrossProcessArtifactPublicationProofResult(
        worker_pids=tuple(worker_pids),
        successes=successes,
        conflicts=conflicts,
        winning_version_id=winning_version_id_observed,
        losing_version_id=losing_version_id_observed,
        final_revision=expected_revision + 1,
        final_current_version_id=winning_version_id_observed,
        history_version_ids=history_version_ids,
    )
