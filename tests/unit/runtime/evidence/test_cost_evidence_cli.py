# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.cli.evidence import run_evidence_cost
from intergrax.runtime.evidence.cost_evidence_contracts import (
    COST_EVIDENCE_REPORT_JSON,
    COST_EVIDENCE_REPORT_MARKDOWN,
    CostEvidenceCheckKind,
    CostEvidenceStatus,
    create_cost_evidence_check_result,
)
from intergrax.runtime.evidence.cost_evidence_runner import build_cost_evidence_report

pytestmark = pytest.mark.unit


def _cost_args(
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


def test_run_evidence_cost_returns_zero_when_passed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_evidence_cost(_cost_args(tmp_path, no_write=True))
    captured = capsys.readouterr()

    assert code == 0
    assert "Cost evidence" in captured.out


def test_run_evidence_cost_default_write_creates_artifacts(tmp_path: Path) -> None:
    out_dir = tmp_path / "cost"
    code = run_evidence_cost(_cost_args(tmp_path, output_dir=out_dir))

    assert code == 0
    assert (out_dir / COST_EVIDENCE_REPORT_JSON).is_file()
    assert (out_dir / COST_EVIDENCE_REPORT_MARKDOWN).is_file()


def test_run_evidence_cost_no_write_does_not_create_files(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "cost"
    code = run_evidence_cost(_cost_args(tmp_path, output_dir=out_dir, no_write=True))
    captured = capsys.readouterr()

    assert code == 0
    assert not out_dir.exists()
    assert "Cost evidence" in captured.out


def test_run_evidence_cost_stdout_contains_report_path_when_writing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_dir = tmp_path / "cost"
    run_evidence_cost(_cost_args(tmp_path, output_dir=out_dir))
    captured = capsys.readouterr()

    assert "report.json" in captured.out


def test_run_evidence_cost_returns_nonzero_when_failed(tmp_path: Path) -> None:
    failed_report = build_cost_evidence_report(
        results=[
            create_cost_evidence_check_result(
                check_id="trace_budget_facets",
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.FAILED,
                title="Trace budget facets unavailable",
            )
        ],
    )
    with patch(
        "intergrax.cli.evidence.run_cost_evidence_checks",
        return_value=failed_report,
    ):
        code = run_evidence_cost(_cost_args(tmp_path, no_write=True))
    assert code != 0


def test_run_evidence_cost_returns_nonzero_when_unavailable(tmp_path: Path) -> None:
    unavailable_report = build_cost_evidence_report(
        results=[
            create_cost_evidence_check_result(
                check_id="trace_budget_facets",
                check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
                status=CostEvidenceStatus.SKIPPED,
                title="Trace budget facets skipped",
            )
        ],
    )
    with patch(
        "intergrax.cli.evidence.run_cost_evidence_checks",
        return_value=unavailable_report,
    ):
        code = run_evidence_cost(_cost_args(tmp_path, no_write=True))
    assert code != 0


def test_evidence_cost_cli_has_no_forbidden_imports() -> None:
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


def test_evidence_cost_cli_does_not_execute_trace_export_or_cli() -> None:
    import inspect

    from intergrax.cli.evidence import run_evidence_cost

    source = inspect.getsource(run_evidence_cost)
    forbidden = (
        "subprocess",
        "runpy",
        "intergrax trace",
        "trace export",
    )
    for token in forbidden:
        assert token not in source, (
            f"run_evidence_cost appears to execute trace export or CLI: {token}"
        )


def test_evidence_cost_cli_does_not_implement_billing_or_provider_pricing() -> None:
    import intergrax.cli.evidence as evidence_module

    forbidden = (
        "stripe",
        "invoice",
        "price_per_token",
        "openai",
        "anthropic",
    )
    path = Path(evidence_module.__file__)
    source = path.read_text(encoding="utf-8").lower()
    for token in forbidden:
        assert token not in source, f"{path.name} contains forbidden token: {token}"
