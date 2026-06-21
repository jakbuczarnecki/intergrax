# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from intergrax.cli.evidence import run_evidence_eval
from intergrax.runtime.evidence.eval_evidence_contracts import (
    EVAL_EVIDENCE_REPORT_JSON,
    EVAL_EVIDENCE_REPORT_MARKDOWN,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _eval_args(
    tmp_path: Path,
    *,
    no_write: bool = False,
    output_dir: Path | None = None,
    root_label: str = "local",
    root: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        root=root if root is not None else tmp_path,
        root_label=root_label,
        output_dir=output_dir,
        no_write=no_write,
    )


def test_run_evidence_eval_returns_zero_when_passed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_evidence_eval(_eval_args(tmp_path, no_write=True, root=_REPO_ROOT))
    captured = capsys.readouterr()

    assert code == 0
    assert "Eval regression evidence" in captured.out


def test_run_evidence_eval_default_write_creates_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval"
    code = run_evidence_eval(
        _eval_args(tmp_path, output_dir=out_dir, root=_REPO_ROOT)
    )

    assert code == 0
    assert (out_dir / EVAL_EVIDENCE_REPORT_JSON).is_file()
    assert (out_dir / EVAL_EVIDENCE_REPORT_MARKDOWN).is_file()


def test_run_evidence_eval_no_write_does_not_create_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "eval"
    code = run_evidence_eval(
        _eval_args(tmp_path, output_dir=out_dir, no_write=True, root=_REPO_ROOT)
    )
    captured = capsys.readouterr()

    assert code == 0
    assert not out_dir.exists()
    assert "Eval regression evidence" in captured.out


def test_run_evidence_eval_stdout_contains_report_path_when_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "eval"
    run_evidence_eval(_eval_args(tmp_path, output_dir=out_dir, root=_REPO_ROOT))
    captured = capsys.readouterr()

    assert "report.json" in captured.out


def test_run_evidence_eval_unavailable_when_source_missing(tmp_path: Path) -> None:
    code = run_evidence_eval(_eval_args(tmp_path, no_write=True))
    assert code != 0


def test_evidence_eval_cli_has_no_forbidden_imports() -> None:
    import intergrax.cli.evidence as evidence_module

    forbidden = (
        "applications.",
        "agents.",
        "from applications",
        "from agents",
        "requests",
        "httpx",
        "urllib",
        "socket",
    )
    path = Path(evidence_module.__file__)
    source = path.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, f"{path.name} contains forbidden import token: {token}"


def test_evidence_eval_cli_does_not_execute_scenario_library_script() -> None:
    import intergrax.cli.evidence as evidence_module

    forbidden = (
        "subprocess",
        "runpy",
        "importlib",
        "check_eval_scenario_library.main",
        " exec(",
        " eval(",
    )
    path = Path(evidence_module.__file__)
    source = path.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"{path.name} appears to execute scenario library script: {token}"
        )
