# © Artur Czarnecki. All rights reserved.

"""Real Mongo recovery qualification for DIAG-FUNCTIONAL-READ-R1-R1."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.document_store import DocumentRecord
from intergrax.integrations.providers.document_store.mongodb.bundle import create_mongodb_document_store
from intergrax.runtime.diagnostics.functional_evidence_execution_index import (
    encode_execution_index_v1,
    encode_execution_index_v2,
    execution_index_v1_row_key,
    execution_index_v2_row_key_from_evidence,
)
from intergrax.runtime.diagnostics.functional_evidence_index_rebuilder import (
    FunctionalEvidenceIndexRebuilder,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence import (
    FunctionalEvidencePersistenceIntegrityError,
)
from intergrax.runtime.diagnostics.functional_evidence_persistence_conformance import (
    collect_all_evidence,
    sample_functional_evidence,
    sample_functional_evidence_scope,
)
from intergrax.runtime.diagnostics.functional_evidence_record_codec import (
    encode_functional_evidence_record,
)
from tests.system.functional_diagnostics_scale.mongodb_backend import resolve_mongodb_uri

_CURSOR_SECRET = b"diag-functional-read-r1r1-qualification-secret"
_PARTITION_PREFIX = "intergrax.functional_evidence.v1"
_EVIDENCE_COUNT = 1000
_INTERRUPT_AFTER_V2_WRITES = 137
_PAGE_SIZE = 25
_BASE_TIME = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ReadR1R1MongoRecoveryResult:
    evidence_count: int
    interrupt_after_v2_writes: int
    rows_written_before_interrupt: int
    recovered_count: int
    page_size: int
    passed: bool


def _partition_key(tenant_id: str) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}"


def _seed_legacy_v1_only(
    *,
    store,
    scope,
    count: int,
) -> None:
    partition_key = _partition_key(scope.tenant_id)
    for index in range(count):
        evidence = sample_functional_evidence(
            scope=scope,
            operation_name=f"legacy-{index}",
            recorded_at=_BASE_TIME + timedelta(seconds=index),
        )
        store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=f"record:{evidence.evidence_id}",
                data=encode_functional_evidence_record(evidence),
            ),
        )
        store.put(
            DocumentRecord(
                partition_key=partition_key,
                row_key=execution_index_v1_row_key(
                    task_id=scope.task_id,
                    run_id=scope.run_id,
                    evidence_id=str(evidence.evidence_id),
                ),
                data=encode_execution_index_v1(str(evidence.evidence_id)),
            ),
        )


def run_read_r1r1_mongo_recovery_qualification(
    *,
    artifact_dir: Path,
) -> ReadR1R1MongoRecoveryResult:
    uri = resolve_mongodb_uri()
    collection = f"diag_functional_read_r1r1_{uuid.uuid4().hex[:12]}"
    inner = assert_conditional_document_store(
        create_mongodb_document_store(
            uri=uri,
            database="intergrax_diag_read_r1r1",
            collection_name=collection,
        ),
    )
    scope = sample_functional_evidence_scope(tenant_id=f"read-r1r1-{uuid.uuid4().hex[:8]}")
    partition_key = _partition_key(scope.tenant_id)
    _seed_legacy_v1_only(store=inner, scope=scope, count=_EVIDENCE_COUNT)

    rows_written = 0

    def _interrupt_after(written: int) -> bool:
        nonlocal rows_written
        rows_written = written
        return written >= _INTERRUPT_AFTER_V2_WRITES

    rebuilder = FunctionalEvidenceIndexRebuilder(
        inner,
        interrupt_after_v2_writes=_interrupt_after,
    )
    try:
        rebuilder.rebuild_execution_index(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
            partition_key=partition_key,
        )
    except FunctionalEvidencePersistenceIntegrityError:
        pass

  # Process B: fresh adapter in subprocess to prove cross-process recovery.
    probe_script = Path(__file__).with_name("recovery_reader_probe.py")
    env = {
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
        "DIAG_R1R1_MONGO_URI": uri,
        "DIAG_R1R1_COLLECTION": collection,
        "DIAG_R1R1_TENANT": scope.tenant_id,
        "DIAG_R1R1_TASK": str(scope.task_id),
        "DIAG_R1R1_RUN": str(scope.run_id),
        "DIAG_R1R1_PAGE_SIZE": str(_PAGE_SIZE),
    }
    completed = subprocess.run(
        [sys.executable, str(probe_script)],
        capture_output=True,
        text=True,
        check=False,
        env={**dict(**__import__("os").environ), **env},
    )
    if completed.returncode != 0:
        inner.close()
        raise RuntimeError(
            f"recovery reader probe failed: {completed.stderr or completed.stdout}",
        )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    recovered_count = int(payload["recovered_count"])

    passed = recovered_count == _EVIDENCE_COUNT
    result = ReadR1R1MongoRecoveryResult(
        evidence_count=_EVIDENCE_COUNT,
        interrupt_after_v2_writes=_INTERRUPT_AFTER_V2_WRITES,
        rows_written_before_interrupt=rows_written,
        recovered_count=recovered_count,
        page_size=_PAGE_SIZE,
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
    outcome = run_read_r1r1_mongo_recovery_qualification(
        artifact_dir=Path(".tmp/proof/diag-functional-read-r1r1"),
    )
    print(json.dumps(asdict(outcome), indent=2))
    raise SystemExit(0 if outcome.passed else 1)
