# © Artur Czarnecki. All rights reserved.

"""Web search qualification plugin adapter (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

import os

from intergrax.core.qualification.functional_diagnostic_expectation import QualificationCaseExpectation
from intergrax.core.qualification.functional_qualification_fidelity import QualificationGateResult, QualificationGateStatus
from intergrax.core.qualification.functional_qualification_identity import WEB_SEARCH_PLUGIN_ID
from intergrax.core.qualification.functional_qualification_plugin import QualificationPluginDescriptor
from intergrax.core.qualification.functional_qualification_result import QualificationPluginResult
from tests.system.functional_diagnostics_q3.cases import (
    MANDATORY_CASES,
    Q3_C_WRONG_SOURCE,
    Q3_G_HEALTHY,
    Q3_G_WRONG_SOURCE,
    _REPEAT_CASE_ID,
)
from tests.system.functional_diagnostics_q3.runner import QualificationReport, QualificationRunRecord, run_qualification
from tests.system.functional_diagnostics_q5.plugins.adapter_common import (
    DomainAdapterConfig,
    DomainQualificationReport,
    DomainRunRecord,
    build_plugin_descriptor,
    build_plugin_result,
    comparison_signature,
)


class WebSearchQualificationPlugin:
    _REPEAT_GROUPS = frozenset({_REPEAT_CASE_ID})
    _HEALTHY = frozenset({"Q3-A", "Q3-G-A"})
    _INCONCLUSIVE = frozenset({"Q3-F"})

    def __init__(self) -> None:
        expectations: dict[str, QualificationCaseExpectation] = {
            case.case_id: case for case in MANDATORY_CASES
        }
        expectations["Q3-G-A"] = Q3_G_HEALTHY
        expectations["Q3-G-B"] = Q3_G_WRONG_SOURCE
        self._expectations = expectations
        self._config = DomainAdapterConfig(
            plugin_id=WEB_SEARCH_PLUGIN_ID,
            domain="web_search",
            display_name="Functional Web Search Qualification",
            artifact_ref=".tmp/session/diag-functional-q5/plugins/functional.web_search.json",
            domain_artifact_ref=".tmp/session/diag-functional-q3/qualification-report.json",
            tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q3_TENANT_ID", "tenant-ue-11g-c1"),
            healthy_case_ids=self._HEALTHY,
            inconclusive_case_ids=self._INCONCLUSIVE,
            expectation_for_record=self._expectation_for_record,
            repeatability_group_ids=self._REPEAT_GROUPS,
            signature_for_record=comparison_signature,
        )

    @property
    def descriptor(self) -> QualificationPluginDescriptor:
        return build_plugin_descriptor(self._config)

    def execute(self) -> QualificationPluginResult:
        report = run_qualification()
        return build_plugin_result(self._config, _adapt_report(report))

    def _expectation_for_record(self, record: DomainRunRecord) -> QualificationCaseExpectation:
        if record.repeat_group == _REPEAT_CASE_ID:
            return Q3_C_WRONG_SOURCE
        return self._expectations.get(record.case_id, Q3_C_WRONG_SOURCE)


def _adapt_report(report: QualificationReport) -> DomainQualificationReport:
    records = tuple(_adapt_record(item) for item in report.records)
    fidelity_pass = all(
        item.evidence_fidelity.candidate_fidelity_match
        and item.evidence_fidelity.selection_fidelity_match
        and item.evidence_fidelity.query_fidelity_match
        and item.evidence_fidelity.extraction_fidelity_match
        and item.evidence_fidelity.identity_fidelity_match
        for item in report.records
        if item.case_id != "Q3-F"
    ) if report.records else False
    selection_fidelity = report.selection_decision_fidelity_percent == 100.0
    selection_authority_fidelity = report.selection_authority_fidelity_percent == 100.0
    extraction_fidelity = report.extraction_decision_fidelity_percent == 100.0
    post_decision_forcing = report.post_decision_forcing == "NONE"
    extra_gates = (
        QualificationGateResult(
            gate_id="selection_decision_fidelity",
            status=QualificationGateStatus.PASS if selection_fidelity else QualificationGateStatus.FAIL,
        ),
        QualificationGateResult(
            gate_id="selection_authority_fidelity",
            status=(
                QualificationGateStatus.PASS
                if selection_authority_fidelity
                else QualificationGateStatus.FAIL
            ),
        ),
        QualificationGateResult(
            gate_id="extraction_decision_fidelity",
            status=QualificationGateStatus.PASS if extraction_fidelity else QualificationGateStatus.FAIL,
        ),
        QualificationGateResult(
            gate_id="post_decision_forcing",
            status=QualificationGateStatus.PASS if post_decision_forcing else QualificationGateStatus.FAIL,
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
        authoritative_attempt_index=record.authoritative_attempt_index,
        attempt_history=record.attempt_history,
        prerequisite_exhausted=record.prerequisite_exhausted,
        blocked_reason=record.blocked_reason,
    )
