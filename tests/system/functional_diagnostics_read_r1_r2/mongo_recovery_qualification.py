# © Artur Czarnecki. All rights reserved.

"""Real Mongo recovery qualification for DIAG-FUNCTIONAL-READ-R1-R2."""

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
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from tests.system.functional_diagnostics_scale.mongodb_backend import resolve_mongodb_uri

_CURSOR_SECRET = b"diag-functional-read-r1r2-qualification-secret"
_BASE_EVIDENCE_COUNT = 1000
_PAGE_SIZE = 25
_BASE_TIME = datetime(2026, 9, 4, 15, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ReadR1R2CrashScenarioResult:
    boundary: str
    expected_count: int
    recovered_count: int
    passed: bool


@dataclass(frozen=True, slots=True)
class ReadR1R2MongoRecoveryResult:
    base_evidence_count: int
    appended_after_crash: int
    expected_count: int
    recovered_count: int
    page_size: int
    crash_matrix: tuple[ReadR1R2CrashScenarioResult, ...]
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


def _run_reader_probe(
    *,
    uri: str,
    collection: str,
    scope,
    expected_count: int,
) -> int:
    probe_script = Path(__file__).with_name("recovery_reader_probe.py")
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "DIAG_R1R2_MONGO_URI": uri,
        "DIAG_R1R2_COLLECTION": collection,
        "DIAG_R1R2_TENANT": scope.tenant_id,
        "DIAG_R1R2_TASK": str(scope.task_id),
        "DIAG_R1R2_RUN": str(scope.run_id),
        "DIAG_R1R2_PAGE_SIZE": str(_PAGE_SIZE),
        "DIAG_R1R2_EXPECTED_COUNT": str(expected_count),
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


def _qualify_crash_matrix(uri: str) -> tuple[ReadR1R2CrashScenarioResult, ...]:
    results: list[ReadR1R2CrashScenarioResult] = []
    for boundary in (
        FunctionalEvidenceAppendFaultBoundary.AFTER_INTENT,
        FunctionalEvidenceAppendFaultBoundary.AFTER_CANONICAL,
        FunctionalEvidenceAppendFaultBoundary.AFTER_V2,
        FunctionalEvidenceAppendFaultBoundary.AFTER_V1,
    ):
        collection = f"diag_functional_read_r1r2_matrix_{boundary.value}_{uuid.uuid4().hex[:8]}"
        inner = assert_conditional_document_store(
            create_mongodb_document_store(
                uri=uri,
                database="intergrax_diag_read_r1r2",
                collection_name=collection,
            ),
        )
        scope = sample_functional_evidence_scope(
            tenant_id=f"r1r2-matrix-{boundary.value}-{uuid.uuid4().hex[:6]}",
        )
        persistence = DocumentStoreFunctionalEvidencePersistence(
            inner,
            cursor_secret=_CURSOR_SECRET,
        )
        _seed_healthy_execution(persistence, scope, 5)
        crashed = sample_functional_evidence(
            scope=scope,
            operation_name=f"crash-{boundary.value}",
            recorded_at=_BASE_TIME + timedelta(seconds=100),
        )
        if boundary is FunctionalEvidenceAppendFaultBoundary.AFTER_INTENT:
            expected = 5
        else:
            expected = 6
        fault = _SingleShotAppendFaultInjector(boundary)
        writer = DocumentStoreFunctionalEvidencePersistence(
            inner,
            cursor_secret=_CURSOR_SECRET,
            append_fault_injector=fault,
        )
        try:
            writer.append(crashed)
        except FunctionalEvidencePersistenceIntegrityError:
            pass
        recovered = _run_reader_probe(
            uri=uri,
            collection=collection,
            scope=scope,
            expected_count=expected,
        )
        results.append(
            ReadR1R2CrashScenarioResult(
                boundary=boundary.value,
                expected_count=expected,
                recovered_count=recovered,
                passed=recovered == expected,
            ),
        )
        inner.close()
    return tuple(results)


def run_read_r1r2_mongo_recovery_qualification(
    *,
    artifact_dir: Path,
) -> ReadR1R2MongoRecoveryResult:
    uri = resolve_mongodb_uri()
    collection = f"diag_functional_read_r1r2_{uuid.uuid4().hex[:12]}"
    inner = assert_conditional_document_store(
        create_mongodb_document_store(
            uri=uri,
            database="intergrax_diag_read_r1r2",
            collection_name=collection,
        ),
    )
    scope = sample_functional_evidence_scope(tenant_id=f"read-r1r2-{uuid.uuid4().hex[:8]}")
    persistence = DocumentStoreFunctionalEvidencePersistence(
        inner,
        cursor_secret=_CURSOR_SECRET,
    )
    _seed_healthy_execution(persistence, scope, _BASE_EVIDENCE_COUNT)
    appended = sample_functional_evidence(
        scope=scope,
        operation_name="post-crash-append",
        recorded_at=_BASE_TIME + timedelta(seconds=_BASE_EVIDENCE_COUNT + 1),
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
    expected_count = _BASE_EVIDENCE_COUNT + 1
    recovered_count = _run_reader_probe(
        uri=uri,
        collection=collection,
        scope=scope,
        expected_count=expected_count,
    )
    crash_matrix = _qualify_crash_matrix(uri)
    passed = recovered_count == expected_count and all(item.passed for item in crash_matrix)
    result = ReadR1R2MongoRecoveryResult(
        base_evidence_count=_BASE_EVIDENCE_COUNT,
        appended_after_crash=1,
        expected_count=expected_count,
        recovered_count=recovered_count,
        page_size=_PAGE_SIZE,
        crash_matrix=crash_matrix,
        passed=passed,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "mongo-recovery-qualification.json").write_text(
        json.dumps(asdict(result), indent=2),
        encoding="utf-8",
    )
    inner.close()
    return result


if __name__ == "__main__":
    outcome = run_read_r1r2_mongo_recovery_qualification(
        artifact_dir=Path(".tmp/proof/diag-functional-read-r1r2"),
    )
    print(json.dumps(asdict(outcome), indent=2))
    raise SystemExit(0 if outcome.passed else 1)
