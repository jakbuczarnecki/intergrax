# © Artur Czarnecki. All rights reserved.

"""Real Mongo qualification for DIAG-FUNCTIONAL-READ-R1-R3 active writer safety."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.runtime.diagnostics.document_store_functional_evidence_persistence import (
    DocumentStoreFunctionalEvidencePersistence,
)
from intergrax.runtime.diagnostics.functional_evidence_append_intent import (
    FunctionalEvidenceAppendFaultBoundary,
    FunctionalEvidenceAppendIntentStore,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from tests.system.functional_diagnostics_scale.mongodb_backend import resolve_mongodb_uri

_CURSOR_SECRET = b"diag-functional-read-r1r3-qualification-secret"
_BASE_EVIDENCE_COUNT = 1000
_PAGE_SIZE = 25
_BASE_TIME = datetime(2026, 9, 4, 16, 30, tzinfo=UTC)
_PARTITION_PREFIX = "intergrax.functional_evidence.v1"


@dataclass(frozen=True, slots=True)
class ReadR1R3ActiveWriterResult:
    reader_fail_closed: bool
    pending_after_reader: bool
    expected_count: int
    recovered_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class ReadR1R3MongoQualificationResult:
    base_evidence_count: int
    active_writer: ReadR1R3ActiveWriterResult
    passed: bool


class _SingleShotAppendFaultInjector:
    def __init__(self, boundary: FunctionalEvidenceAppendFaultBoundary) -> None:
        self._boundary = boundary
        self._fired = False

    def should_fault_after(self, boundary: FunctionalEvidenceAppendFaultBoundary) -> bool:
        if not self._fired and boundary is self._boundary:
            self._fired = True
            return True
        return False


def _seed_healthy_execution(
    persistence: DocumentStoreFunctionalEvidencePersistence,
    scope,
    count: int,
) -> None:
    for index in range(count):
        persistence.append(
            sample_functional_evidence(
                scope=scope,
                operation_name=f"base-{index}",
                recorded_at=_BASE_TIME + timedelta(seconds=index),
            ),
        )


def _run_consistency_pending_probe(*, uri: str, collection: str, scope) -> dict[str, object]:
    probe_script = Path(__file__).with_name("consistency_pending_reader_probe.py")
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "DIAG_R1R3_MONGO_URI": uri,
        "DIAG_R1R3_COLLECTION": collection,
        "DIAG_R1R3_TENANT": scope.tenant_id,
        "DIAG_R1R3_TASK": str(scope.task_id),
        "DIAG_R1R3_RUN": str(scope.run_id),
    }
    completed = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**__import__("os").environ), **env},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"consistency pending probe failed: {completed.stderr or completed.stdout}",
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def _run_recovery_probe(
    *,
    uri: str,
    collection: str,
    scope,
    expected_count: int,
) -> int:
    probe_script = Path(__file__).with_name("recovery_reader_probe.py")
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "DIAG_R1R3_MONGO_URI": uri,
        "DIAG_R1R3_COLLECTION": collection,
        "DIAG_R1R3_TENANT": scope.tenant_id,
        "DIAG_R1R3_TASK": str(scope.task_id),
        "DIAG_R1R3_RUN": str(scope.run_id),
        "DIAG_R1R3_PAGE_SIZE": str(_PAGE_SIZE),
        "DIAG_R1R3_EXPECTED_COUNT": str(expected_count),
    }
    completed = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**__import__("os").environ), **env},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"recovery reader probe failed: {completed.stderr or completed.stdout}",
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return int(payload["recovered_count"])


def run_read_r1r3_mongo_qualification(
    *,
    artifact_dir: Path,
) -> ReadR1R3MongoQualificationResult:
    uri = resolve_mongodb_uri()
    collection = f"diag_functional_read_r1r3_{uuid.uuid4().hex[:12]}"
    inner = assert_conditional_document_store(
        create_mongodb_document_store(
            uri=uri,
            database="intergrax_diag_read_r1r3",
            collection_name=collection,
        ),
    )
    scope = sample_functional_evidence_scope(tenant_id=f"read-r1r3-{uuid.uuid4().hex[:8]}")
    partition_key = f"{_PARTITION_PREFIX}:{scope.tenant_id}"
    persistence = DocumentStoreFunctionalEvidencePersistence(
        inner,
        cursor_secret=_CURSOR_SECRET,
    )
    _seed_healthy_execution(persistence, scope, _BASE_EVIDENCE_COUNT)
    appended = sample_functional_evidence(
        scope=scope,
        operation_name="active-writer-append",
        recorded_at=_BASE_TIME + timedelta(seconds=_BASE_EVIDENCE_COUNT + 1),
    )
    intent_store = FunctionalEvidenceAppendIntentStore(inner)
    intent_store.create_pending(
        partition_key=partition_key,
        task_id=scope.task_id,
        run_id=scope.run_id,
        evidence_id=str(appended.evidence_id),
    )
    inner.close()

    reader_payload = _run_consistency_pending_probe(
        uri=uri,
        collection=collection,
        scope=scope,
    )
    reader_fail_closed = bool(reader_payload["consistency_pending"])
    pending_after_reader = bool(reader_payload["pending_exists"])

    inner = assert_conditional_document_store(
        create_mongodb_document_store(
            uri=uri,
            database="intergrax_diag_read_r1r3",
            collection_name=collection,
        ),
    )
    fault = _SingleShotAppendFaultInjector(FunctionalEvidenceAppendFaultBoundary.AFTER_CANONICAL)
    writer = DocumentStoreFunctionalEvidencePersistence(
        inner,
        cursor_secret=_CURSOR_SECRET,
        append_fault_injector=fault,
    )
    try:
        writer.append(appended)
    except FunctionalEvidencePersistenceIntegrityError:
        pass
    inner.close()

    expected_count = _BASE_EVIDENCE_COUNT + 1
    recovered_count = _run_recovery_probe(
        uri=uri,
        collection=collection,
        scope=scope,
        expected_count=expected_count,
    )
    active_writer = ReadR1R3ActiveWriterResult(
        reader_fail_closed=reader_fail_closed,
        pending_after_reader=pending_after_reader,
        expected_count=expected_count,
        recovered_count=recovered_count,
        passed=(
            reader_fail_closed
            and pending_after_reader
            and recovered_count == expected_count
        ),
    )
    result = ReadR1R3MongoQualificationResult(
        base_evidence_count=_BASE_EVIDENCE_COUNT,
        active_writer=active_writer,
        passed=active_writer.passed,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "mongo-active-writer-qualification.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    return result


if __name__ == "__main__":
    outcome = run_read_r1r3_mongo_qualification(
        artifact_dir=Path(".tmp/proof/diag-functional-read-r1r3"),
    )
    print(json.dumps(asdict(outcome), indent=2))
    raise SystemExit(0 if outcome.passed else 1)
