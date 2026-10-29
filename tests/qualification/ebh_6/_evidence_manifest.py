# © Artur Czarnecki. All rights reserved.

"""Evidence SHA manifest for EBH-6 dossier and mechanical gates."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Final

EBH6_EVIDENCE_SUITE: Final[str] = "tests/qualification/ebh_6/"

# Operator planning anchor — EBH-6 wave START_HEAD @ approved execution.
EXPECTED_START_HEAD: Final[str] = "cc76a4df1f401c8d7c374ec3b8d2bd7f21461b97"

EBH5_INDEPENDENT_AUDIT_SHA: Final[str] = "33ecf19cc020b112d9e586c87b49e819202f5d6d"
CONFIG_X_HISTORICAL_PASS_SHA: Final[str] = "fac9a43f3b128fde63541d2b56ebd8f4a751e8fb"
EBH5_AUDIT_BASELINE_SHA: Final[str] = "83a376b3e1b312e2e4f88da548c99c328f3589f7"

# Qualification artifact publication commit (P0–P6 tree); HEAD may advance for doc-only SHA ledger updates.
EBH6_PUBLICATION_SHA: Final[str] = "86d4d1a4c029be469b561a137f0fdedc2c8fd942"

# Final enterprise closure wave — substance commit; doc-only descendants remain valid @ qual.
EBH6_CERTIFICATION_HANDOFF_SHA: Final[str] = "c852e7705f318b5d50a771596d97193f68a00e67"


@lru_cache(maxsize=1)
def evidence_git_head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def implementation_sha() -> str:
    return evidence_git_head_sha()


def qualification_sha() -> str:
    return evidence_git_head_sha()


@lru_cache(maxsize=1)
def git_object_exists(sha: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-t", sha],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def expected_start_head_resolvable() -> bool:
    return git_object_exists(EXPECTED_START_HEAD)


def assert_ebh5_audit_baseline_resolvable() -> None:
    assert git_object_exists(EBH5_INDEPENDENT_AUDIT_SHA), "EBH-5 independent audit SHA must resolve"
    assert git_object_exists(CONFIG_X_HISTORICAL_PASS_SHA), "CONFIG-X historical PASS SHA must resolve"
