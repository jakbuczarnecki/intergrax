# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from platform_proofs.tools.iterative_sql_investigation.dataset_identity import PROOF_ID
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ToolsSqlInvestigationProofResult,
)

DEFAULT_ARTIFACT_ROOT = Path(".artifacts") / "proof" / PROOF_ID
PROOF_RESULT_FILENAME = "proof-result.json"


def resolve_artifact_root(explicit: Path | None = None) -> Path:
    """Resolve the proof-local artifact root under the current working directory."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (Path.cwd() / DEFAULT_ARTIFACT_ROOT).resolve()


def allocate_run_directory(
    *,
    artifact_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Create a run-scoped subdirectory for generated proof artifacts."""
    root = resolve_artifact_root(artifact_root)
    resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_directory = root / resolved_run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def write_proof_result(
    result: ToolsSqlInvestigationProofResult,
    *,
    run_directory: Path,
) -> Path:
    """Persist the machine-readable proof result JSON under the run directory."""
    path = run_directory / PROOF_RESULT_FILENAME
    path.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
