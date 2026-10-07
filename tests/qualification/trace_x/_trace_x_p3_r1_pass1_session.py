# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P3-R1-R1 Pass 1 session evidence recorder (no nested pytest)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.qualification.trace_x._trace_x_p3_r1_support import (
    PASS1_MECHANICAL_NODEIDS,
    normalize_pytest_nodeid,
)

PASS1_PASSED_NODEIDS: set[str] = set()

PASS1_OBSERVED_MANIFEST = Path(".tmp/session/trace-x-p3-r1-r1-q3/pass1_observed_nodeids.json")


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
        return
    PASS1_OBSERVED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    PASS1_OBSERVED_MANIFEST.write_text(
        json.dumps(sorted(PASS1_PASSED_NODEIDS)),
        encoding="utf-8",
    )
    _assert_pass1_enterprise_audit_closure()


def load_pass1_observed_nodeids() -> set[str]:
    if PASS1_OBSERVED_MANIFEST.is_file():
        return set(json.loads(PASS1_OBSERVED_MANIFEST.read_text(encoding="utf-8")))
    return set(PASS1_PASSED_NODEIDS)


def _assert_pass1_enterprise_audit_closure() -> None:
    from tests.qualification.trace_x._trace_x_p3_r1_support import (
        ENTERPRISE_AUDIT_MATRIX_GATE_IDS,
        ENTERPRISE_AUDIT_MATRIX_P3_R1,
        ENTERPRISE_AUDIT_MATRIX_ROW_GATES,
        P3_R1_R1_GATE_BY_ID,
        P3_R1_R1_Q_BLK_01_RESOLUTION,
        R1GateResult,
        TENANT_ISOLATION_AUDIT_P3_R1_R1,
        observed_audit_row_result,
        observed_gate_passed,
    )

    passed = {normalize_pytest_nodeid(nodeid) for nodeid in PASS1_PASSED_NODEIDS}
    cross_tenant = normalize_pytest_nodeid(TENANT_ISOLATION_AUDIT_P3_R1_R1["cross_tenant_path"])
    assert cross_tenant in passed
    assert observed_gate_passed("TXP3R1R1-Q17", passed)
    for row in ENTERPRISE_AUDIT_MATRIX_P3_R1:
        observed = observed_audit_row_result(row.area, passed, pass1_only=True)
        assert observed is R1GateResult.PASS, (
            f"{row.area} gates={ENTERPRISE_AUDIT_MATRIX_ROW_GATES[row.area]} observed={observed}"
        )
        if row.area in ENTERPRISE_AUDIT_MATRIX_GATE_IDS:
            for gate_id in ENTERPRISE_AUDIT_MATRIX_GATE_IDS[row.area]:
                evidence = P3_R1_R1_GATE_BY_ID[gate_id]
                if not evidence.pass1_required or gate_id == "TXP3R1R1-Q30":
                    continue
                assert observed_gate_passed(gate_id, passed), gate_id
    assert P3_R1_R1_Q_BLK_01_RESOLUTION.startswith("RESOLVED")
