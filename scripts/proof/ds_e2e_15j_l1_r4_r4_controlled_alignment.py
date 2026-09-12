# © Artur Czarnecki. All rights reserved.

"""CLI: DS-E2E-15J-L1.R4.R4 controlled MODEL_OVERCOMMIT qualification proof."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from testing_support.decision_e2e.controlled_alignment.artifacts import (
    QualificationStatus,
    default_artifact_dir,
    write_qualification_artifacts,
)
from testing_support.decision_e2e.controlled_alignment.runner import (
    run_model_overcommit_controlled_qualification,
)
from testing_support.decision_e2e.controlled_alignment.source_freeze import (
    TASK_ID,
    verify_controlled_alignment_source_freeze,
    write_source_freeze_baseline,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=TASK_ID)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--write-source-freeze-baseline",
        action="store_true",
        help="Capture semantic source freeze baseline (maintainer only).",
    )
    args = parser.parse_args(argv)

    if args.write_source_freeze_baseline:
        path = write_source_freeze_baseline(args.repo_root)
        print(f"baseline_written={path}")
        return 0

    freeze = verify_controlled_alignment_source_freeze(args.repo_root)
    print(f"SOURCE_FREEZE_STATUS={freeze.status.value}")
    if freeze.status is not SourceFreezeStatus.PASS:
        return 2

    run_result = asyncio.run(run_model_overcommit_controlled_qualification())
    artifact = write_qualification_artifacts(
        args.repo_root,
        run_result,
        output_dir=args.output_dir or default_artifact_dir(args.repo_root),
    )
    print(f"STATUS={artifact.status.value}")
    print(f"REPAIR={run_result.repair_evidence.repair_status.value}")
    print(f"output_dir={artifact.output_dir}")
    return 0 if artifact.status is QualificationStatus.PASS else 3


if __name__ == "__main__":
    sys.exit(main())
