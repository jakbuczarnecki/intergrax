# © Artur Czarnecki. All rights reserved.

"""Evidence SHA manifest for EBH-5 dossier."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Final

EBH5_EVIDENCE_SUITE: Final[str] = "tests/qualification/ebh_5/"

# Parent wave commit audited by Protocol 03 (corrective batch baseline).
AUDIT_BASELINE_SHA: Final[str] = "83a376b3e1b312e2e4f88da548c99c328f3589f7"


@lru_cache(maxsize=1)
def evidence_git_head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def proof_execution_sha() -> str:
    """SHA under which EBH-5 qualification tests were last executed (current HEAD at run time)."""
    return evidence_git_head_sha()


@lru_cache(maxsize=1)
def git_object_exists(sha: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-t", sha],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def assert_audit_baseline_resolvable() -> None:
    assert len(AUDIT_BASELINE_SHA) == 40
    assert git_object_exists(AUDIT_BASELINE_SHA), "AUDIT_BASELINE_SHA must resolve in git object database"
