# © Artur Czarnecki. All rights reserved.

"""ROADMAP-REPLAY-X L2 corrective wave — scoped production delta dispositions @ 91649e5 publication lineage."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPLAY_X_L2_QUALIFICATION_HEAD: Final[str] = "91649e5d883664a855c02dea1baf14d467052fca"
_DUP_X_PIN = "a7d83925b375b5f08eb0f175ced2c17fa026bba9"


@dataclass(frozen=True, slots=True)
class ProdDeltaDisposition:
    path: str
    disposition: str
    regression_evidence: str
    qualification_head_sha: str = REPLAY_X_L2_QUALIFICATION_HEAD


REPLAY_X_L2_SCOPED_PROD_DELTA: Final[tuple[ProdDeltaDisposition, ...]] = (
    ProdDeltaDisposition(
        path="intergrax/runtime/resilience/dependency_attempt_execution_boundary.py",
        disposition="CTRL-X pyright callback typing + ruff format; admission/shutdown semantics unchanged",
        regression_evidence="tests/qualification/control_planes/ctrl_x/test_ctrl_x_typing_gates.py",
    ),
)

REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS: Final[frozenset[str]] = frozenset(
    row.path for row in REPLAY_X_L2_SCOPED_PROD_DELTA
)


def _prod_diff_paths(repo: Path, *roots: str) -> frozenset[str]:
    names = subprocess.check_output(
        ["git", "diff", "--name-only", f"{_DUP_X_PIN}..HEAD", "--", *roots],
        cwd=repo,
        text=True,
    ).splitlines()
    return frozenset(n.strip().replace("\\", "/") for n in names if n.strip())


def scoped_prod_delta_disposition_violations(repo: Path) -> tuple[str, ...]:
    violations: list[str] = []
    prod_diff = _prod_diff_paths(repo, "intergrax", "agents", "applications")
    for row in REPLAY_X_L2_SCOPED_PROD_DELTA:
        if row.path in prod_diff:
            evidence = repo / row.regression_evidence
            if not evidence.is_file():
                violations.append(f"missing_regression_evidence:{row.path}")
            else:
                text = evidence.read_text(encoding="utf-8")
                if "def test_" not in text:
                    violations.append(f"invalid_regression_contract:{row.regression_evidence}")
            if row.qualification_head_sha != REPLAY_X_L2_QUALIFICATION_HEAD:
                violations.append(f"wrong_qualification_head:{row.path}")
            if not row.disposition.strip():
                violations.append(f"missing_disposition:{row.path}")
        elif row.path in REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS:
            violations.append(f"scoped_path_not_in_prod_diff:{row.path}")
    for changed in prod_diff:
        if changed not in REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS:
            continue
    return tuple(violations)


def historical_evidence_invalidation_violations(repo: Path) -> tuple[str, ...]:
    violations = list(scoped_prod_delta_disposition_violations(repo))
    for root in ("intergrax", "agents", "applications"):
        tier_diff = _prod_diff_paths(repo, root)
        unscoped = [p for p in tier_diff if p not in REPLAY_X_L2_SCOPED_PROD_DELTA_PATHS]
        if unscoped:
            violations.append(f"unscoped_prod_delta:{root}:{unscoped[0]}")
    return tuple(violations)
