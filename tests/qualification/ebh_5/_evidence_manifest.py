# © Artur Czarnecki. All rights reserved.

"""Evidence SHA manifest for EBH-5 dossier."""

from __future__ import annotations

import subprocess
from functools import lru_cache
from typing import Final

EBH5_EVIDENCE_SUITE: Final[str] = "tests/qualification/ebh_5/"


@lru_cache(maxsize=1)
def evidence_git_head_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()
