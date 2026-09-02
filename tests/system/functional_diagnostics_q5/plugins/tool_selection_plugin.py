# © Artur Czarnecki. All rights reserved.

"""Tool selection qualification plugin adapter (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

import os

from intergrax.core.qualification.functional_diagnostic_expectation import QualificationCaseExpectation
from intergrax.core.qualification.functional_qualification_fidelity import QualificationGateResult, QualificationGateStatus
from intergrax.core.qualification.functional_qualification_identity import TOOL_SELECTION_PLUGIN_ID
from intergrax.core.qualification.functional_qualification_plugin import QualificationPluginDescriptor
from intergrax.core.qualification.functional_qualification_result import QualificationPluginResult
from tests.system.functional_diagnostics_q2.cases import (
    MANDATORY_CASES,
    Q2_B_WRONG_TOOL,
    Q2_F_HEALTHY,
    Q2_F_WRONG_TOOL,
    _REPEAT_CASE_ID,
)
from tests.system.functional_diagnostics_q2.runner import QualificationReport, QualificationRunRecord, run_qualification
from tests.system.functional_diagnostics_q5.plugins.adapter_common import (
    DomainAdapterConfig,
    DomainQualificationReport,
    DomainRunRecord,
    build_plugin_descriptor,
    build_plugin_result,
    comparison_signature,
)


class ToolSelectionQualificationPlugin:
    _REPEAT_GROUPS = frozenset({_REPEAT_CASE_ID})
    _HEALTHY = frozenset({"Q2-A", "Q2-F-A"})
    _INCONCLUSIVE = frozenset({"Q2-E"})

    def __init__(self) -> None:
        expectations: dict[str, QualificationCaseExpectation] = {
            case.case_id: case for case in MANDATORY_CASES
        }
        expectations["Q2-F-A"] = Q2_F_HEALTHY
        expectations["Q2-F-B"] = Q2_F_WRONG_TOOL
        self._expectations = expectations
        self._config = DomainAdapterConfig(
            plugin_id=TOOL_SELECTION_PLUGIN_ID,
            domain="tool_selection",
            display_name="Functional Tool Selection Qualification",
            artifact_ref=".tmp/session/diag-functional-q5/plugins/functional.tool_selection.json",
            domain_artifact_ref=".tmp/session/diag-functional-q2/qualification-report.json",
            tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q2_TENANT_ID", "tenant-ue-11g-c1"),
            healthy_case_ids=self._HEALTHY,
            inconclusive_case_ids=self._INCONCLUSIVE,
            expectation_for_record=self._expectation_for_record,
            repeatability_group_ids=self._REPEAT_GROUPS,
            signature_for_record=_tool_signature,
        )

    @property
    def descriptor(self) -> QualificationPluginDescriptor:
        return build_plugin_descriptor(self._config)

    def execute(self) -> QualificationPluginResult:
        report = run_qualification()
        return build_plugin_result(self._config, _adapt_report(report))

    def _expectation_for_record(self, record: DomainRunRecord) -> QualificationCaseExpectation:
        if record.repeat_group == _REPEAT_CASE_ID:
            return Q2_B_WRONG_TOOL
        return self._expectations.get(record.case_id, Q2_B_WRONG_TOOL)


def _tool_signature(record: DomainRunRecord) -> tuple[str, ...]:
    base = comparison_signature(record)
    return base


def _adapt_report(report: QualificationReport) -> DomainQualificationReport:
    records = tuple(_adapt_record(item) for item in report.records)
    fidelity_pass = all(
        item.evidence_fidelity.candidate_fidelity_match
        and item.evidence_fidelity.selection_fidelity_match
        and item.evidence_fidelity.invocation_fidelity_match
        and item.evidence_fidelity.identity_fidelity_match
        for item in report.records
        if item.case_id != "Q2-E"
    ) if report.records else False
    extra_gates = (
        QualificationGateResult(
            gate_id="invocation_fidelity",
            status=QualificationGateStatus.PASS if fidelity_pass else QualificationGateStatus.FAIL,
        ),
    )
    return DomainQualificationReport(
        verdict=report.verdict,
        records=records,
        blocked_reason=report.blocked_reason,
        repeatability_pass=report.repeatability_pass,
        fidelity_pass=fidelity_pass,
        decision_independence_pass=True,
        domain_extra_gates=extra_gates,
    )


def _adapt_record(record: QualificationRunRecord) -> DomainRunRecord:
    fidelity = record.evidence_fidelity
    return DomainRunRecord(
        case_id=record.case_id,
        task_id=record.task_id,
        run_id=record.run_id,
        comparison=record.comparison,
        functional_outcome=record.functional_outcome,
        diag_first_failed_check=record.diag_first_failed_check,
        operator_outcome=record.operator_outcome,
        repeat_group=record.repeat_group,
        identity_fidelity_match=fidelity.identity_fidelity_match,
    )
