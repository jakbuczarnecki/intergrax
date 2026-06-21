# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.evidence import run_evidence_live_core
from intergrax.runtime.evidence.live_core_probe_export import (
    LIVE_CORE_PROBE_REPORT_JSON,
    LIVE_CORE_PROBE_REPORT_MARKDOWN,
)

pytestmark = pytest.mark.unit


def _live_core_args(
    tmp_path: Path,
    *,
    no_write: bool = False,
    output_dir: Path | None = None,
    root_label: str = "local",
) -> argparse.Namespace:
    return argparse.Namespace(
        root=tmp_path,
        root_label=root_label,
        output_dir=output_dir,
        no_write=no_write,
    )


def test_run_evidence_live_core_returns_zero_when_passed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_evidence_live_core(_live_core_args(tmp_path, no_write=True))
    captured = capsys.readouterr()

    assert code == 0
    assert "Selected live Tier-0 probes" in captured.out


def test_run_evidence_live_core_default_write_creates_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "live_core_probes"
    code = run_evidence_live_core(_live_core_args(tmp_path, output_dir=out_dir))

    assert code == 0
    assert (out_dir / LIVE_CORE_PROBE_REPORT_JSON).is_file()
    assert (out_dir / LIVE_CORE_PROBE_REPORT_MARKDOWN).is_file()


def test_run_evidence_live_core_no_write_does_not_create_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "live_core_probes"
    code = run_evidence_live_core(
        _live_core_args(tmp_path, output_dir=out_dir, no_write=True)
    )
    captured = capsys.readouterr()

    assert code == 0
    assert not out_dir.exists()
    assert "Selected live Tier-0 probes" in captured.out


def test_run_evidence_live_core_stdout_contains_report_path_when_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "live_core_probes"
    run_evidence_live_core(_live_core_args(tmp_path, output_dir=out_dir))
    captured = capsys.readouterr()

    assert "live_core_report.json" in captured.out


def test_evidence_live_core_cli_has_no_forbidden_imports() -> None:
    import intergrax.cli.evidence as evidence_module

    forbidden = ("applications.", "agents.", "from applications", "from agents")
    path = Path(evidence_module.__file__)
    source = path.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, f"{path.name} contains forbidden import token: {token}"
