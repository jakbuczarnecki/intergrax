# © Artur Czarnecki. All rights reserved.

"""Git-derived production delta evidence for TENANT-X-FINAL-R1 (committed objects only)."""

from __future__ import annotations

import subprocess
from typing import Final

from tests.qualification.tenant_x._final_types import FinalProductionDeltaEvidence
from tests.qualification.tenant_x._final_wave_evidence import current_git_head_sha

TENANT_X_FINAL_REJECTED_BASE_SHA: Final[str] = "940e6219ecf95d3202236a51b2fa284403583e5f"
TENANT_X_FINAL_R1_REJECTED_BASE_SHA: Final[str] = "036dcbd732e01a412a2cd86a81f6c1b040c9c2ec"

_PRODUCTION_PREFIXES: Final[tuple[str, ...]] = (
    "intergrax/",
    "agents/",
    "applications/",
)


def _is_production_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in _PRODUCTION_PREFIXES)


def build_final_production_delta_evidence(
    *,
    base_sha: str = TENANT_X_FINAL_REJECTED_BASE_SHA,
    head_sha: str | None = None,
) -> FinalProductionDeltaEvidence | None:
    head = head_sha if head_sha is not None else current_git_head_sha()
    if len(base_sha) != 40 or len(head) != 40:
        return None
    try:
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{base_sha}^{{commit}}"],
            stderr=subprocess.DEVNULL,
        )
        subprocess.check_output(
            ["git", "cat-file", "-e", f"{head}^{{commit}}"],
            stderr=subprocess.DEVNULL,
        )
        raw = subprocess.check_output(
            ["git", "diff", "--name-only", f"{base_sha}..{head}"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    changed = tuple(sorted({line.strip() for line in raw.splitlines() if line.strip() and _is_production_path(line.strip())}))
    return FinalProductionDeltaEvidence(
        base_sha=base_sha,
        head_sha=head,
        changed_production_paths=changed,
        production_delta_count=len(changed),
    )
