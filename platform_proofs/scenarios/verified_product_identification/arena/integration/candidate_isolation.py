"""Isolated subprocess execution for single arena candidates."""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path
from typing import Literal

from platform_proofs.scenarios.verified_product_identification.arena.contracts.execution_budget import (
    EmbeddingArenaExecutionBudget,
)

CandidatePhase = Literal["stage_ab", "stage_c"]

_CANDIDATE_MODULE = (
    "platform_proofs.scenarios.verified_product_identification.run_embedding_arena_candidate"
)


def _candidate_artifact_path(
    session_dir: Path,
    candidate_id: str,
    phase: CandidatePhase,
) -> Path:
    return session_dir / "candidates" / candidate_id / f"{phase}.pkl"


def write_candidate_phase_artifact(
    session_dir: Path,
    candidate_id: str,
    phase: CandidatePhase,
    payload: object,
) -> Path:
    path = _candidate_artifact_path(session_dir, candidate_id, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)
    return path


def load_candidate_phase_artifact(
    session_dir: Path,
    candidate_id: str,
    phase: CandidatePhase,
) -> object | None:
    path = _candidate_artifact_path(session_dir, candidate_id, phase)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def run_candidate_phase_subprocess(
    *,
    candidate_id: str,
    phase: CandidatePhase,
    execution_budget: EmbeddingArenaExecutionBudget,
    session_dir: Path,
    include_e5_control: bool,
) -> subprocess.CompletedProcess[str] | None:
    session_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        _CANDIDATE_MODULE,
        "--candidate-id",
        candidate_id,
        "--profile",
        execution_budget.profile_id,
        "--phase",
        phase,
        "--session-dir",
        str(session_dir),
    ]
    if include_e5_control:
        command.append("--include-e5-control")
    try:
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=execution_budget.candidate_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
