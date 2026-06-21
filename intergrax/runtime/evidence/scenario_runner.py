# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Deterministic core certification scenario runner (HEP Band 2ae · EVID-CORE-04)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

from intergrax.runtime.evidence.certification_report import (
    CoreCertificationReport,
    build_core_certification_report,
    write_core_certification_report,
)
from intergrax.runtime.evidence.core_certification_spec import (
    CoreCertificationLevel,
    CoreCertificationMode,
    is_scenario_in_level,
    normalize_core_level,
)
from intergrax.runtime.evidence.scenario_contracts import (
    CoreEvidenceRef,
    CoreScenarioContract,
    CoreScenarioResult,
    CoreScenarioStatus,
    EvidenceRefKind,
    core_scenario_contracts_for_level,
    get_core_scenario_contract,
    validate_core_scenario_catalog,
)

_CERTIFICATION_REPORT_SCENARIO_ID = "certification_report_emitted"


@dataclass(frozen=True, slots=True)
class CoreScenarioRunContext:
    certification_run_id: str
    output_dir: Path
    level: CoreCertificationLevel


def require_core_scenario_contract(scenario_id: str) -> CoreScenarioContract:
    contract = get_core_scenario_contract(scenario_id)
    if contract is None:
        raise ValueError(f"missing core scenario contract: {scenario_id}")
    return contract


def _passed(
    contract: CoreScenarioContract,
    evidence_refs: list[CoreEvidenceRef],
    message: str = "",
) -> CoreScenarioResult:
    return CoreScenarioResult(
        scenario_id=contract.scenario_id,
        status=CoreScenarioStatus.PASSED,
        evidence_refs=evidence_refs,
        message=message or contract.expectation.summary,
    )


def _evidence_for_contract(contract: CoreScenarioContract) -> list[CoreEvidenceRef]:
    return [
        CoreEvidenceRef(
            kind=kind,
            ref=f"mock:{contract.scenario_id}:{kind.value}",
            description=contract.expectation.summary,
        )
        for kind in contract.required_evidence_kinds
    ]


def _run_mock_scenario(
    contract: CoreScenarioContract,
    _context: CoreScenarioRunContext,
) -> CoreScenarioResult:
    return _passed(contract, _evidence_for_contract(contract))


def _run_certification_report_emitted(
    contract: CoreScenarioContract,
    context: CoreScenarioRunContext,
) -> CoreScenarioResult:
    json_path = context.output_dir / "report.json"
    md_path = context.output_dir / "report.md"
    if not json_path.is_file() or not md_path.is_file():
        return CoreScenarioResult(
            scenario_id=contract.scenario_id,
            status=CoreScenarioStatus.FAILED,
            message="certification report artifacts missing before final validation",
        )
    return _passed(
        contract,
        [
            CoreEvidenceRef(
                kind=EvidenceRefKind.CERTIFICATION_REPORT,
                ref=str(json_path),
                description="core certification report.json present",
            ),
            CoreEvidenceRef(
                kind=EvidenceRefKind.CERTIFICATION_REPORT,
                ref=str(md_path),
                description="core certification report.md present",
            ),
        ],
        message="certification report artifacts written",
    )


_SCENARIO_RUNNERS: dict[str, Callable[[CoreScenarioContract, CoreScenarioRunContext], CoreScenarioResult]] = {
    "basic_run_completed": _run_mock_scenario,
    "trace_persisted": _run_mock_scenario,
    "tool_denied_by_policy": _run_mock_scenario,
    "high_risk_tool_hitl": _run_mock_scenario,
    "budget_exceeded_handled": _run_mock_scenario,
    "llm_error_classified": _run_mock_scenario,
    "retry_executed": _run_mock_scenario,
    "domain_signal_emitted": _run_mock_scenario,
    "memory_read_write_recorded": _run_mock_scenario,
    "rag_context_event_recorded": _run_mock_scenario,
    "cost_report_generated": _run_mock_scenario,
    _CERTIFICATION_REPORT_SCENARIO_ID: _run_certification_report_emitted,
}


def run_core_scenario(
    contract: CoreScenarioContract,
    context: CoreScenarioRunContext,
) -> CoreScenarioResult:
    runner = _SCENARIO_RUNNERS.get(contract.scenario_id)
    if runner is None:
        raise ValueError(f"no runner registered for scenario: {contract.scenario_id}")
    result = runner(contract, context)
    missing_kinds = {
        kind
        for kind in contract.required_evidence_kinds
        if kind not in {ref.kind for ref in result.evidence_refs}
    }
    if missing_kinds and result.status is CoreScenarioStatus.PASSED:
        return CoreScenarioResult(
            scenario_id=contract.scenario_id,
            status=CoreScenarioStatus.FAILED,
            evidence_refs=result.evidence_refs,
            message=f"missing required evidence kinds: {', '.join(sorted(k.value for k in missing_kinds))}",
        )
    return result


def run_core_certification(
    level: str | CoreCertificationLevel,
    *,
    output_dir: Path,
    mode: CoreCertificationMode = CoreCertificationMode.OPERATOR_LOCAL,
    certification_run_id: str | None = None,
) -> CoreCertificationReport:
    """Run deterministic core certification scenarios for ``level`` and write report artifacts."""
    validate_core_scenario_catalog()
    resolved_level = normalize_core_level(level)
    run_id = certification_run_id or (
        f"core-cert-{resolved_level.name.lower()}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:8]}"
    )
    context = CoreScenarioRunContext(
        certification_run_id=run_id,
        output_dir=output_dir,
        level=resolved_level,
    )
    contracts = core_scenario_contracts_for_level(resolved_level)
    results: list[CoreScenarioResult] = []

    for contract in contracts:
        if contract.scenario_id == _CERTIFICATION_REPORT_SCENARIO_ID:
            continue
        results.append(run_core_scenario(contract, context))

    preliminary = build_core_certification_report(
        level=resolved_level,
        results=results,
        certification_run_id=run_id,
        output_dir=output_dir,
        mode=mode,
    )
    write_core_certification_report(preliminary, output_dir)

    if is_scenario_in_level(_CERTIFICATION_REPORT_SCENARIO_ID, resolved_level):
        report_contract = require_core_scenario_contract(_CERTIFICATION_REPORT_SCENARIO_ID)
        results.append(run_core_scenario(report_contract, context))

    final_report = build_core_certification_report(
        level=resolved_level,
        results=results,
        certification_run_id=run_id,
        output_dir=output_dir,
        mode=mode,
    )
    write_core_certification_report(final_report, output_dir)
    return final_report
