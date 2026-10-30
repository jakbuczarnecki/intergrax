# © Artur Czarnecki. All rights reserved.

"""Current-HEAD VBN replay failures — seven artifacts + mechanical register (B6-05)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from tests.qualification.ebh_6._findings_register import count_prohibited_unresolved

_REPO = Path(__file__).resolve().parents[3]
_DUP_X_PIN = "a7d83925b375b5f08eb0f175ced2c17fa026bba9"

REPLAY_VBN_ARTIFACT_REL_PATHS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CORE_HARDCODING_FINDINGS_REGISTER.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CROSS_LAYER_REPLACEABILITY_MATRIX.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_VENDOR_SELECTION_REACHABILITY_EVIDENCE.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md",
)


def _production_effective_vendor_delta_paths(repo: Path) -> tuple[str, ...]:
    names = subprocess.check_output(
        ["git", "diff", "--name-only", f"{_DUP_X_PIN}..HEAD", "--", "intergrax", "agents", "applications"],
        cwd=repo,
        text=True,
    ).splitlines()
    return tuple(n.strip().replace("\\", "/") for n in names if n.strip())


def vbn_replay_failure_violations(
    repo: Path | None = None,
    *,
    prohibited_unresolved_override: int | None = None,
) -> tuple[str, ...]:
    root = _REPO if repo is None else repo
    violations: list[str] = []
    for rel in REPLAY_VBN_ARTIFACT_REL_PATHS:
        path = root / rel
        if not path.is_file():
            violations.append(f"missing_vbn_artifact:{rel}")
            continue
        if len(path.read_text(encoding="utf-8").strip()) < 80:
            violations.append(f"empty_vbn_artifact:{rel}")
    prohibited = (
        count_prohibited_unresolved() if prohibited_unresolved_override is None else prohibited_unresolved_override
    )
    if prohibited:
        violations.append(f"prohibited_unresolved:{prohibited}")
    core_md = root / REPLAY_VBN_ARTIFACT_REL_PATHS[1]
    if core_md.is_file():
        text = core_md.read_text(encoding="utf-8")
        match = re.search(r"PROHIBITED unresolved[^\d]*(\d+)", text, re.IGNORECASE)
        if match and int(match.group(1)) != prohibited:
            violations.append("vbn_narrative_vs_register_mismatch")
    unscoped = [
        p
        for p in _production_effective_vendor_delta_paths(root)
        if p != "intergrax/runtime/resilience/dependency_attempt_execution_boundary.py"
    ]
    if unscoped:
        violations.append(f"unexpected_prod_delta_for_vbn:{unscoped[0]}")
    return tuple(violations)


def count_vendor_neutrality_replay_failures(
    repo: Path | None = None,
    *,
    prohibited_unresolved_override: int | None = None,
) -> int:
    return len(
        vbn_replay_failure_violations(
            repo,
            prohibited_unresolved_override=prohibited_unresolved_override,
        )
    )
