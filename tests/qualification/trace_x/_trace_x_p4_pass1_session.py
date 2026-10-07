# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R3 Pass 1 session evidence recorder."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p4_support import (
    P4_R4_GATE_REGISTRY,
    gate_nodeids,
    nodeid_observed,
    normalize_gate_nodeid,
    observed_gate_passed,
)

PASS1_PASSED_NODEIDS: set[str] = set()

PASS1_OBSERVED_MANIFEST = Path(".tmp/session/trace-x-p4-r4/pass1_observed_nodeids.json")


def normalize_pytest_nodeid(nodeid: str) -> str:
    return normalize_gate_nodeid(nodeid)


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if os.environ.get("TRACE_X_P4_PASS1") != "1":
        return
    if report.when != "call" or not report.passed:
        return
    PASS1_PASSED_NODEIDS.add(normalize_pytest_nodeid(report.nodeid))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if os.environ.get("TRACE_X_P4_PASS1") != "1":
        return
    collected_files = {item.nodeid.split("::", 1)[0].rsplit("/", 1)[-1] for item in session.items}
    if "test_trace_x_p4_r4_closed_world.py" not in collected_files:
        return
    required_nodeids = gate_nodeids(P4_R4_GATE_REGISTRY)
    collected = {normalize_pytest_nodeid(item.nodeid) for item in session.items}
    missing = [nid for nid in required_nodeids if not nodeid_observed(nid, PASS1_PASSED_NODEIDS)]
    if missing:
        session.exitstatus = 1
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line(
                "TRACE-X-P4-R3 Pass 1 missing mechanical nodeids: " + ", ".join(sorted(missing)),
                red=True,
            )
        return
    PASS1_OBSERVED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PASS1_OBSERVED_MANIFEST.write_text(json.dumps(sorted(PASS1_PASSED_NODEIDS)), encoding="utf-8")
    for row in P4_R4_GATE_REGISTRY:
        if not row.pass1_required:
            continue
        assert observed_gate_passed(row.gate_id, PASS1_PASSED_NODEIDS), row.gate_id
