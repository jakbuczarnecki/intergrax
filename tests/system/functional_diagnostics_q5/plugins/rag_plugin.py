# © Artur Czarnecki. All rights reserved.

"""RAG qualification plugin adapter (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

import os

from intergrax.core.qualification.functional_diagnostic_expectation import QualificationCaseExpectation
from intergrax.core.qualification.functional_qualification_identity import (
    MODEL_ROUTING_PLUGIN_ID,
    RAG_PLUGIN_ID,
    TOOL_SELECTION_PLUGIN_ID,
    WEB_SEARCH_PLUGIN_ID,
)
from intergrax.core.qualification.functional_qualification_plugin import (
    FunctionalQualificationPlugin,
    QualificationPluginDescriptor,
)
from intergrax.core.qualification.functional_qualification_result import QualificationPluginResult
from tests.system.functional_diagnostics_q1.cases import (
    MANDATORY_CASES,
    Q1_B_SELECTION_FAILURE,
    Q1_C_SYNTHESIS_FAILURE,
    Q1_E_FAILURE,
    Q1_E_HEALTHY,
    Q1_H_HISTORICAL_WRONG_DATE,
)
from tests.system.functional_diagnostics_q1.runner import QualificationReport, QualificationRunRecord, run_qualification
from tests.system.functional_diagnostics_q5.plugins.adapter_common import (
    DomainAdapterConfig,
    DomainQualificationReport,
    DomainRunRecord,
    build_plugin_descriptor,
    build_plugin_result,
    comparison_signature,
)


class RagQualificationPlugin:
    _REPEAT_GROUPS = frozenset({"Q1-F", "Q1-C-R"})
    _HEALTHY = frozenset({"Q1-A", "Q1-E-A", "Q1-H"})
    _INCONCLUSIVE = frozenset({"Q1-D"})

    def __init__(self) -> None:
        expectations: dict[str, QualificationCaseExpectation] = {
            case.case_id: case for case in MANDATORY_CASES
        }
        expectations["Q1-E-A"] = Q1_E_HEALTHY
        expectations["Q1-E-B"] = Q1_E_FAILURE
        expectations["Q1-H"] = Q1_H_HISTORICAL_WRONG_DATE
        self._expectations = expectations
        self._config = DomainAdapterConfig(
            plugin_id=RAG_PLUGIN_ID,
            domain="rag",
            display_name="Functional RAG Qualification",
            artifact_ref=".tmp/session/diag-functional-q5/plugins/functional.rag.json",
            domain_artifact_ref=".tmp/session/diag-functional-q1/qualification-report.json",
            tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q1_TENANT_ID", "tenant-ue-11g-c1"),
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
        if record.repeat_group == "Q1-F":
            return Q1_B_SELECTION_FAILURE
        if record.repeat_group == "Q1-C-R":
            return Q1_C_SYNTHESIS_FAILURE
        return self._expectations.get(record.case_id, Q1_B_SELECTION_FAILURE)


def _adapt_report(report: QualificationReport) -> DomainQualificationReport:
    records = tuple(_adapt_record(item) for item in report.records)
    fidelity_pass = all(
        item.evidence_fidelity.candidate_fidelity_match
        and item.evidence_fidelity.selection_fidelity_match
        and item.evidence_fidelity.identity_fidelity_match
        for item in report.records
        if item.case_id != "Q1-D"
    ) if report.records else False
    return DomainQualificationReport(
        verdict=report.verdict,
        records=records,
        blocked_reason=report.blocked_reason,
        repeatability_pass=report.repeatability_pass,
        fidelity_pass=fidelity_pass,
        decision_independence_pass=True,
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
