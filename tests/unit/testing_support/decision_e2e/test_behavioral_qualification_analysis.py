# © Artur Czarnecki. All rights reserved.

"""DS-E2E-15J-L1.R4.R1.ANALYSIS pipeline smoke tests."""

from __future__ import annotations

from pathlib import Path

from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_analysis import (
    run_behavioral_analysis,
)
from testing_support.decision_e2e.local_qualification_session.behavioral_qualification_source_freeze import (
    SourceFreezeStatus,
)


def test_r4r1_behavioral_analysis_reproducible(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[4]
    session_dir = (
        repo_root / ".tmp" / "session" / "DS-E2E-15J-L1-R4-R1" / "cohort-retry-dirty"
    )
    if not (session_dir / "runs.json").is_file():
        return
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    first = run_behavioral_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=out_a,
    )
    second = run_behavioral_analysis(
        repo_root=repo_root,
        session_dir=session_dir,
        output_dir=out_b,
    )
    assert first.source_freeze.status is SourceFreezeStatus.PASS
    assert first.alignment.total_runs == second.alignment.total_runs == 20
    assert (out_a / "artifact-manifest.txt").is_file()
    assert (out_b / "final-analysis-report.md").read_text(
        encoding="utf-8"
    ) == (out_a / "final-analysis-report.md").read_text(encoding="utf-8")
