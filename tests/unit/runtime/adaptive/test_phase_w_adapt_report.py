# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.12: phase_w_adapt_report script tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal
from intergrax.runtime.adaptive.signal_store import SQLiteSignalStore
from scripts.release.phase_w_adapt_report import build_signal_trend_report

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_build_signal_trend_report_aggregates_utilities(tmp_path) -> None:
    db_path = tmp_path / "signals.db"
    store = SQLiteSignalStore(db_path=db_path)
    store.append(
        HarnessOutcomeSignal(
            run_id="run_a",
            tenant_id="t1",
            application_id="lab",
            agent_id="echo",
            task_class="echo.basic",
            utility=0.8,
        )
    )
    store.append(
        HarnessOutcomeSignal(
            run_id="run_b",
            tenant_id="t1",
            application_id="lab",
            agent_id="echo",
            task_class="echo.basic",
            utility=0.4,
        )
    )
    report = build_signal_trend_report(store)
    assert report.signal_count == 2
    assert report.average_utility == pytest.approx(0.6)
    assert report.tenant_ids == ["t1"]


def test_phase_w_adapt_report_cli_writes_json(tmp_path) -> None:
    db_path = tmp_path / "signals.db"
    store = SQLiteSignalStore(db_path=db_path)
    store.append(
        HarnessOutcomeSignal(
            run_id="run_cli",
            tenant_id="t-cli",
            application_id="lab",
            agent_id="echo",
            task_class="echo.basic",
            utility=0.55,
        )
    )
    output = tmp_path / "signal_trends.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "release" / "phase_w_adapt_report.py"),
            "--db-path",
            str(db_path),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["signal_count"] == 1
    assert payload["average_utility"] == pytest.approx(0.55)
