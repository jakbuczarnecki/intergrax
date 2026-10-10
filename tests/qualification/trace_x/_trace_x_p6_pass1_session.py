# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P6 adversarial bundle Pass 1 session evidence recorder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p6_adversarial_matrix import P6_ADVERSARIAL_MATRIX

PASS1_PASSED_NODEIDS: set[str] = set()

PASS1_OBSERVED_MANIFEST = Path(
    ".tmp/session/trace-x-p6/pass1_observed_nodeids.json",
)


def _test_id_observed(passed: set[str], test_id: str) -> bool:
    normalized_test_id = test_id.removeprefix("test_")
    for entry in passed:
        leaf = entry.replace("\\", "/").rsplit("::", 1)[-1]
        base = leaf.split("[", 1)[0]
        if base == test_id or base.removeprefix("test_") == normalized_test_id:
            return True
    return False


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if os.environ.get("TRACE_X_P6_PASS1") != "1":
        return
    if report.when != "call" or not report.passed:
        return
    PASS1_PASSED_NODEIDS.add(report.nodeid.replace("\\", "/"))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("TRACE_X_P6_PASS1") != "1":
        return
    collected_files = {
        item.nodeid.split("::", 1)[0].rsplit("/", 1)[-1] for item in session.items
    }
    if "test_trace_x_p6_adversarial_bundle.py" not in collected_files:
        return
    missing = [
        row.test_id
        for row in P6_ADVERSARIAL_MATRIX
        if not _test_id_observed(PASS1_PASSED_NODEIDS, row.test_id)
    ]
    if missing:
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "TRACE-X-P6 Pass 1 missing adversarial nodeids: "
                + ", ".join(sorted(missing)),
                red=True,
            )
        return
    PASS1_OBSERVED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PASS1_OBSERVED_MANIFEST.write_text(
        json.dumps(sorted(PASS1_PASSED_NODEIDS)),
        encoding="utf-8",
    )
