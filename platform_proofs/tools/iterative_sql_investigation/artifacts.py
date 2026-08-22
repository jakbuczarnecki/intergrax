# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-3C).

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from scripts.proof.intergrax_platform_proof_evidence import PlatformProofEvidence
from scripts.proof.intergrax_platform_proof_execution import INTERGRAX_PROOF_ARTIFACT_DIR_ENV
from scripts.proof.intergrax_platform_proof_evidence_io import (
    EVIDENCE_FILENAME,
    write_evidence_json,
)

from platform_proofs.tools.iterative_sql_investigation.report_renderer import (
    write_tools_sql_investigation_report,
)

from platform_proofs.tools.iterative_sql_investigation.dataset_identity import PROOF_ID
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ToolsSqlInvestigationProofResult,
)

_PACKAGE_ROOT = Path(__file__).resolve().parent
CANONICAL_PACKAGE_OUTPUT = _PACKAGE_ROOT / "output"
DEFAULT_ARTIFACT_ROOT = Path(".artifacts") / "proof" / PROOF_ID
PROOF_RESULT_FILENAME = "proof-result.json"


def canonical_package_output_directory() -> Path:
    """Return the commit-ready canonical output directory for this proof package."""
    return CANONICAL_PACKAGE_OUTPUT.resolve()


def resolve_artifact_root(explicit: Path | None = None) -> Path:
    """Resolve the proof-local artifact root under the current working directory."""
    if explicit is not None:
        return explicit.expanduser().resolve()
    return (Path.cwd() / DEFAULT_ARTIFACT_ROOT).resolve()


def resolve_runner_artifact_directory() -> Path | None:
    """Return runner-provided proof artifact directory when suite execution set it."""
    raw = os.environ.get(INTERGRAX_PROOF_ARTIFACT_DIR_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def allocate_run_directory(
    *,
    artifact_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    """Create the directory for generated proof artifacts."""
    runner_directory = resolve_runner_artifact_directory()
    if runner_directory is not None:
        return runner_directory
    if artifact_root is not None or run_id is not None:
        root = resolve_artifact_root(artifact_root)
        resolved_run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_directory = root / resolved_run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory
    run_directory = canonical_package_output_directory()
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


def write_evidence(
    evidence: PlatformProofEvidence,
    *,
    run_directory: Path,
) -> Path:
    """Persist generic Platform Proof evidence JSON under the run directory."""
    return write_evidence_json(evidence, proof_directory=run_directory)


def write_report(
    evidence: PlatformProofEvidence,
    *,
    run_directory: Path,
) -> Path:
    """Persist self-contained HTML report from the same typed evidence instance."""
    return write_tools_sql_investigation_report(evidence, run_directory=run_directory)
