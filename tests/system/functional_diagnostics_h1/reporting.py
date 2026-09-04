# © Artur Czarnecki. All rights reserved.

"""Machine and human report serialization for DIAG-FUNCTIONAL-H1."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    DiagnosticHealthReport,
    GateResult,
    HealthGateId,
    HealthVerdict,
    H1_SCHEMA_VERSION,
    H1_SEMANTICS,
)


def _gate_to_json(gate: GateResult) -> dict[str, str | tuple[str, ...]]:
    return {
        "gate_id": gate.gate_id.value,
        "verdict": gate.verdict.value,
        "summary": gate.summary,
        "details": gate.details,
    }


def health_report_to_json(report: DiagnosticHealthReport) -> dict[str, object]:
    return {
        "schema_version": report.schema_version,
        "tested_sha": report.tested_sha,
        "start_head": report.start_head,
        "final_head": report.final_head,
        "timestamp": report.timestamp,
        "h1_semantics": report.h1_semantics,
        "inventory_counts": report.inventory_counts,
        "collection_result": _gate_to_json(report.collection_result),
        "static_results": _gate_to_json(report.static_results),
        "unit_results": _gate_to_json(report.unit_results),
        "repeatability_results": [asdict(item) for item in report.repeatability_results],
        "local_system_results": _gate_to_json(report.local_system_results),
        "external_preflight_results": [asdict(item) for item in report.external_preflight_results],
        "skip_xfail_inventory": [asdict(item) for item in report.skip_xfail_inventory],
        "invariant_coverage": [asdict(item) for item in report.invariant_coverage],
        "dead_stale_findings": report.dead_stale_findings,
        "gate_results": [_gate_to_json(item) for item in report.gate_results],
        "core_test_health": report.core_test_health.value,
        "real_service_qualification_availability": report.real_service_qualification_availability.value,
        "overall_h1": report.overall_h1.value,
        "blocking_findings": report.blocking_findings,
        "warnings": report.warnings,
    }


def write_health_report(path: Path, report: DiagnosticHealthReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = health_report_to_json(report)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def write_test_inventory(path: Path, inventory: tuple[object, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(item) for item in inventory]
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def aggregate_overall_verdict(
    gate_results: tuple[GateResult, ...],
    *,
    real_service_blocked: bool,
) -> tuple[HealthVerdict, HealthVerdict, HealthVerdict]:
    core_gates = {
        HealthGateId.H1_A_COLLECTION,
        HealthGateId.H1_B_CORE_HEALTH,
        HealthGateId.H1_C_REPEATABILITY,
        HealthGateId.H1_D_INVARIANT_COVERAGE,
        HealthGateId.H1_E_SKIP_XFAIL_HONESTY,
        HealthGateId.H1_F_EXTERNAL_DEPENDENCY,
        HealthGateId.H1_G_RUNNER_INTEGRITY,
        HealthGateId.H1_H_STALE_DEAD,
        HealthGateId.H1_I_SUPERSESSION,
        HealthGateId.H1_J_REPORT_INTEGRITY,
    }
    core_failed = any(
        gate.verdict is HealthVerdict.FAILED for gate in gate_results if gate.gate_id in core_gates
    )
    core_verdict = HealthVerdict.FAILED if core_failed else HealthVerdict.PASS
    real_service = HealthVerdict.BLOCKED if real_service_blocked else HealthVerdict.PASS
    overall = core_verdict
    return core_verdict, real_service, overall


def build_human_report(report: DiagnosticHealthReport) -> str:
    lines = [
        "# DIAG-FUNCTIONAL-H1 TEST-SUITE HEALTH QUALIFICATION",
        "",
        "## Verdict",
        report.overall_h1.value,
        "",
        "## Start HEAD",
        report.start_head,
        "",
        "## Final HEAD",
        report.final_head,
        "",
        "## Qualified SHA",
        report.tested_sha,
        "",
        "## Scope",
        "Diagnostic Engine test-suite health (inventory, gates, repeatability, invariant ownership).",
        "",
        "## H1 semantics",
        report.h1_semantics,
        "",
        "## Test inventory",
        "",
    ]
    for layer, count in sorted(report.inventory_counts.items()):
        lines.append(f"- {layer}: {count}")
    lines.extend(
        [
            "",
            "## H1-A collection",
            f"{report.collection_result.verdict.value} — {report.collection_result.summary}",
            "",
            "## H1-B core health",
            f"{report.unit_results.verdict.value} — {report.unit_results.summary}",
            "",
            "## H1-C repeatability",
        ]
    )
    for run in report.repeatability_results:
        lines.append(
            f"- {run.scope}: collected={run.collected} passed={run.passed} failed={run.failed} verdict={run.verdict.value}"
        )
    lines.extend(
        [
            "",
            "## H1-D architecture invariant coverage",
            f"{next(g for g in report.gate_results if g.gate_id.value == 'H1-D').verdict.value}",
            "",
            "## H1-E skip/xfail audit",
            f"findings={len(report.skip_xfail_inventory)}",
            "",
            "## H1-F external dependency classification",
        ]
    )
    for item in report.external_preflight_results:
        lines.append(f"- {item.family.value}: {item.state.value} ({item.note})")
    lines.extend(
        [
            "",
            "## H1-G qualification runner integrity",
            next(g for g in report.gate_results if g.gate_id.value == "H1-G").summary,
            "",
            "## H1-H stale/dead tests",
            "NONE" if not report.dead_stale_findings else "\n".join(report.dead_stale_findings),
            "",
            "## H1-I supersession consistency",
            next(g for g in report.gate_results if g.gate_id.value == "H1-I").summary,
            "",
            "## H1-J machine report integrity",
            next(g for g in report.gate_results if g.gate_id.value == "H1-J").summary,
            "",
            "## Core vs real-service",
            f"CORE_TEST_HEALTH={report.core_test_health.value}",
            f"REAL_SERVICE_QUALIFICATION_AVAILABILITY={report.real_service_qualification_availability.value}",
            "",
            "## Blocking findings",
            "NONE" if not report.blocking_findings else "\n".join(f"- {item}" for item in report.blocking_findings),
            "",
            "## Machine artifact",
            ".tmp/session/diag-functional-h1/qualification-report.json",
            "",
            "## Final architecture statement",
        ]
    )
    if report.core_test_health is HealthVerdict.PASS:
        lines.extend(
            [
                "DIAGNOSTIC TEST-SUITE HEALTH = QUALIFIED",
                "CRITICAL DIAGNOSTIC INVARIANTS = OWNED BY EXECUTABLE TEST GATES",
                "DETERMINISTIC CORE DIAGNOSTIC TESTS = REPEATABLE",
                "EXTERNAL SERVICE ABSENCE = EXPLICITLY BLOCKED, NEVER FALSE PASS",
                "HISTORICAL QUALIFICATION != CURRENT LIVE REVALIDATION",
            ]
        )
    return "\n".join(lines) + "\n"


def utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def new_report_shell(*, start_head: str, tested_sha: str) -> DiagnosticHealthReport:
    placeholder = GateResult(
        gate_id=HealthGateId.H1_J_REPORT_INTEGRITY,
        verdict=HealthVerdict.FAILED,
        summary="pending",
    )
    return DiagnosticHealthReport(
        schema_version=H1_SCHEMA_VERSION,
        tested_sha=tested_sha,
        start_head=start_head,
        final_head=start_head,
        timestamp=utc_now_iso(),
        h1_semantics=H1_SEMANTICS,
        inventory_counts={},
        collection_result=placeholder,
        static_results=placeholder,
        unit_results=placeholder,
        repeatability_results=(),
        local_system_results=placeholder,
        external_preflight_results=(),
        skip_xfail_inventory=(),
        invariant_coverage=(),
        dead_stale_findings=(),
        gate_results=(),
        core_test_health=HealthVerdict.FAILED,
        real_service_qualification_availability=HealthVerdict.BLOCKED,
        overall_h1=HealthVerdict.FAILED,
        blocking_findings=(),
        warnings=(),
    )
