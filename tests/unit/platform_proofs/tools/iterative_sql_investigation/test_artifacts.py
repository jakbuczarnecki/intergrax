# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

from platform_proofs.tools.iterative_sql_investigation.artifacts import (
    PROOF_RESULT_FILENAME,
    allocate_run_directory,
    resolve_artifact_root,
    write_proof_result,
)
from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    DatasetIdentity,
    compute_dataset_fingerprint,
)
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ToolsSqlInvestigationProofResult,
)


def test_resolve_artifact_root_uses_explicit_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "proof-artifacts"
    explicit.mkdir()

    assert resolve_artifact_root(explicit) == explicit.resolve()


def test_write_proof_result_persists_json_under_run_directory(tmp_path: Path) -> None:
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)
    result = ToolsSqlInvestigationProofResult.blocked(
        proof_id="TOOLS-ITERATIVE-SQL-INVESTIGATION",
        identity=identity,
        fingerprint=fingerprint,
        reason="test",
    )
    run_directory = allocate_run_directory(
        artifact_root=tmp_path,
        run_id="test-run",
    )

    artifact_path = write_proof_result(result, run_directory=run_directory)

    assert artifact_path == run_directory / PROOF_RESULT_FILENAME
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["blocked_reason"] == "test"
    assert payload["proof_id"] == "TOOLS-ITERATIVE-SQL-INVESTIGATION"
