# © Artur Czarnecki. All rights reserved.

"""Accepted TENANT-X wave evidence SHAs and current-HEAD ancestry checks."""

from __future__ import annotations

import subprocess
from typing import Final

from tests.qualification.tenant_x._final_types import AcceptedWaveEvidence, TenantXWave

ACCEPTED_TENANT_X_WAVE_EVIDENCE: Final[tuple[AcceptedWaveEvidence, ...]] = (
    AcceptedWaveEvidence(
        wave=TenantXWave.P0,
        accepted_evidence_sha="3b1a86b138750a1ab289ac328151feecb9f7496c",
    ),
    AcceptedWaveEvidence(
        wave=TenantXWave.R1,
        accepted_evidence_sha="220fd6e9c5c59fb7bae6c789ebee0484d8bc7c0e",
    ),
    AcceptedWaveEvidence(
        wave=TenantXWave.R2,
        accepted_evidence_sha="9bc53dfad65dc20d327d7c7340953264c5d69c7a",
    ),
    AcceptedWaveEvidence(
        wave=TenantXWave.R3,
        accepted_evidence_sha="c3e708febf6a138258f65ca0fa391925fa3e2ad8",
    ),
    AcceptedWaveEvidence(
        wave=TenantXWave.R4,
        accepted_evidence_sha="d903a8dfaf7acaa53b013e2f7074fde9814441d2",
        rejected_lineage_shas=(
            "b57ba7cfef45e7fac699c83792ebe432be19879d",
            "eec0d0ef3b77b53ca49cbbb8aada1bf7dcb97b7c",
        ),
    ),
)


def current_git_head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_is_ancestor(ancestor_sha: str, descendant_sha: str | None = None) -> bool:
    head = descendant_sha if descendant_sha is not None else current_git_head_sha()
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", ancestor_sha, head],
        check=False,
    )
    return proc.returncode == 0


def accepted_wave_ancestry_violations(head_sha: str | None = None) -> tuple[str, ...]:
    head = head_sha if head_sha is not None else current_git_head_sha()
    violations: list[str] = []
    for item in ACCEPTED_TENANT_X_WAVE_EVIDENCE:
        if item.wave == TenantXWave.FINAL:
            continue
        if not git_is_ancestor(item.accepted_evidence_sha, head):
            violations.append(f"{item.wave.value}:accepted_sha_not_ancestor_of_head")
    return tuple(violations)


def accepted_wave_by_id(wave: TenantXWave) -> AcceptedWaveEvidence:
    for item in ACCEPTED_TENANT_X_WAVE_EVIDENCE:
        if item.wave == wave:
            return item
    raise KeyError(wave)
