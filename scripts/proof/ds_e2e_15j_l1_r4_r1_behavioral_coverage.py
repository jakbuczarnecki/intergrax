# © Artur Czarnecki. All rights reserved.

"""CLI: DS-E2E-15J-L1.R4.R1.OBS behavioral evidence coverage analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.behavioral_coverage_analysis import (
    OBS_TASK_ID,
    run_behavioral_coverage_analysis,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)


def _default_session_dir(repo_root: Path) -> Path:
    candidates = (
        repo_root
        / ".artifacts"
        / "qualification"
        / "DS-E2E-15J-L1.R4.R1",
        repo_root / ".tmp" / "session" / "DS-E2E-15J-L1-R4-R1" / "cohort-retry-dirty",
    )
    for parent in candidates:
        if parent.is_dir():
            if (parent / "runs.json").is_file():
                return parent
            for child in sorted(parent.iterdir()):
                if child.is_dir() and (child / "runs.json").is_file():
                    return child
    return candidates[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=OBS_TASK_ID)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root
        / ".artifacts"
        / "qualification"
        / "DS-E2E-15J-L1.R4.R1.OBS",
    )
    args = parser.parse_args(argv)
    session_dir = args.session_dir or _default_session_dir(args.repo_root)
    result = run_behavioral_coverage_analysis(
        repo_root=args.repo_root,
        session_dir=session_dir,
        output_dir=args.output_dir,
    )
    print(f"OBS_SOURCE_FREEZE_STATUS={result.source_freeze.status.value}")
    print(f"status={result.final_analysis_status.value}")
    print(f"15K_B_EFFECT={result.fifteen_kb_effect.value}")
    print(f"output_dir={result.output_dir}")
    if result.source_freeze.status is not SourceFreezeStatus.PASS:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
