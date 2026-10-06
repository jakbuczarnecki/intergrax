# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1-R1 Pass 1 session evidence recorder (no nested pytest)."""

from __future__ import annotations

import os

import pytest

from tests.qualification.trace_x._trace_x_p3_r1_support import (
    PASS1_MECHANICAL_NODEIDS,
    normalize_pytest_nodeid,
)

PASS1_PASSED_NODEIDS: set[str] = set()


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if os.environ.get("TRACE_X_P3_R1_R1_PASS1") != "1":
        return
    if report.when != "call" or not report.passed:
        return
    PASS1_PASSED_NODEIDS.add(normalize_pytest_nodeid(report.nodeid))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("TRACE_X_P3_R1_R1_PASS1") != "1":
        return
    required = {
        normalize_pytest_nodeid(nodeid) for nodeid in PASS1_MECHANICAL_NODEIDS
    }
    collected = {normalize_pytest_nodeid(item.nodeid) for item in session.items}
    if not required.intersection(collected):
        return
    missing = required - PASS1_PASSED_NODEIDS
    if missing:
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "TRACE-X-P3-R1-R1 Pass 1 missing mechanical nodeids: "
                + ", ".join(sorted(missing)),
                red=True,
            )
