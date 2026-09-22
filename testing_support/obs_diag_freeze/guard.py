# © Artur Czarnecki. All rights reserved.

"""OBS/DIAG frozen boundary change-control guard (qualification scope, not runtime)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from testing_support.obs_diag_freeze.manifest import (
    CERTIFICATION_METADATA_RECONCILIATION_COMMIT,
    CERTIFICATION_RECORD_COMMIT,
    CERTIFIED_CODE_SHA,
    FROZEN_CONTRACT_MODULES,
    OBS_DIAG_FROZEN_REUSED_GATE_MODULES,
    STALE_CERTIFICATION_RECORD_COMMIT,
)


@dataclass(frozen=True, slots=True)
class ProvenanceCheckResult:
    certified_code_is_ancestor: bool
    certification_record_is_ancestor: bool
    metadata_reconciliation_is_ancestor: bool


def git_is_ancestor(repo_root: Path, ancestor: str, descendant: str = "HEAD") -> bool:
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def verify_certification_provenance(repo_root: Path) -> ProvenanceCheckResult:
    return ProvenanceCheckResult(
        certified_code_is_ancestor=git_is_ancestor(repo_root, CERTIFIED_CODE_SHA),
        certification_record_is_ancestor=git_is_ancestor(repo_root, CERTIFICATION_RECORD_COMMIT),
        metadata_reconciliation_is_ancestor=git_is_ancestor(
            repo_root,
            CERTIFICATION_METADATA_RECONCILIATION_COMMIT,
        ),
    )


def missing_frozen_contract_modules(repo_root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    for rel in FROZEN_CONTRACT_MODULES:
        if not (repo_root / rel).is_file():
            missing.append(rel)
    return tuple(missing)


def stale_certification_record_referenced_in_freeze_support(repo_root: Path) -> tuple[str, ...]:
    support_root = repo_root / "testing_support" / "obs_diag_freeze"
    hits: list[str] = []
    for path in support_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in {".py", ".md"}:
            continue
        if path.name == "manifest.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if STALE_CERTIFICATION_RECORD_COMMIT in text:
            hits.append(path.relative_to(repo_root).as_posix())
    return tuple(hits)


def missing_reused_gate_modules(repo_root: Path) -> tuple[str, ...]:
    missing: list[str] = []
    for rel in OBS_DIAG_FROZEN_REUSED_GATE_MODULES:
        if not (repo_root / rel).is_file():
            missing.append(rel)
    return tuple(missing)


def run_pytest_modules(repo_root: Path, modules: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "pytest", *modules, "-q", "--tb=short"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
