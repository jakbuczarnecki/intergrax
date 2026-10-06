# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R2 Pass 1 session evidence recorder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p4_support import P4_R2_GATE_REGISTRY, gate_nodeids, observed_gate_passed

PASS1_PASSED_NODEIDS: set[str] = set()

PASS1_OBSERVED_MANIFEST = Path(".tmp/session/trace-x-p4-r2/pass1_observed_nodeids.json")


def normalize_pytest_nodeid(nodeid: str) -> str:
    return nodeid.split("::")[-1]


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if os.environ.get("TRACE_X_P4_PASS1") != "1":
        return
    if report.when != "call" or not report.passed:
        return
    PASS1_PASSED_NODEIDS.add(normalize_pytest_nodeid(report.nodeid))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("TRACE_X_P4_PASS1") != "1":
        return
    required = {normalize_pytest_nodeid(nodeid) for nodeid in gate_nodeids(P4_R2_GATE_REGISTRY)}
    collected = {normalize_pytest_nodeid(item.nodeid) for item in session.items}
    if not required.intersection(collected):
        return
    missing = required - PASS1_PASSED_NODEIDS
    if missing:
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "TRACE-X-P4-R2 Pass 1 missing mechanical nodeids: " + ", ".join(sorted(missing)),
                red=True,
            )
        return
    PASS1_OBSERVED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PASS1_OBSERVED_MANIFEST.write_text(json.dumps(sorted(PASS1_PASSED_NODEIDS)), encoding="utf-8")
    for row in P4_R2_GATE_REGISTRY:
        if not row.pass1_required:
            continue
        assert observed_gate_passed(row.gate_id, PASS1_PASSED_NODEIDS), row.gate_id
