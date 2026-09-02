# © Artur Czarnecki. All rights reserved.

"""Model routing qualification plugin adapter (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

import os

from intergrax.core.qualification.functional_diagnostic_expectation import QualificationCaseExpectation
from intergrax.core.qualification.functional_qualification_fidelity import QualificationGateResult, QualificationGateStatus
from intergrax.core.qualification.functional_qualification_identity import MODEL_ROUTING_PLUGIN_ID
from intergrax.core.qualification.functional_qualification_plugin import QualificationPluginDescriptor
from intergrax.core.qualification.functional_qualification_result import QualificationPluginResult
from tests.system.functional_diagnostics_q4.cases import (
    MANDATORY_CASES,
    Q4_B_WRONG_ROUTE,
    Q4_F_HEALTHY,
    Q4_F_WRONG_ROUTE,
    _REPEAT_CASE_ID,
)
from tests.system.functional_diagnostics_q4.runner import QualificationReport, QualificationRunRecord, run_qualification
from tests.system.functional_diagnostics_q5.plugins.adapter_common import (
    DomainAdapterConfig,
    DomainQualificationReport,
    DomainRunRecord,
    build_plugin_descriptor,
    build_plugin_result,
    comparison_signature,
)


class ModelRoutingQualificationPlugin:
    _REPEAT_GROUPS = frozenset({_REPEAT_CASE_ID})
    _HEALTHY = frozenset({"Q4-A", "Q4-F-A"})
    _INCONCLUSIVE = frozenset({"Q4-E"})

    def __init__(self) -> None:
        expectations: dict[str, QualificationCaseExpectation] = {
            case.case_id: case for case in MANDATORY_CASES
        }
        expectations["Q4-F-A"] = Q4_F_HEALTHY
        expectations["Q4-F-B"] = Q4_F_WRONG_ROUTE
        self._expectations = expectations
        self._config = DomainAdapterConfig(
            plugin_id=MODEL_ROUTING_PLUGIN_ID,
            domain="model_routing",
            display_name="Functional Model Routing Qualification",
            artifact_ref=".tmp/session/diag-functional-q5/plugins/functional.model_routing.json",
            domain_artifact_ref=".tmp/session/diag-functional-q4/qualification-report.json",
            tenant_id=os.environ.get("DIAG_FUNCTIONAL_Q4_TENANT_ID", "tenant-ue-11g-c1"),
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
            return Q4_B_WRONG_ROUTE
        return self._expectations.get(record.case_id, Q4_B_WRONG_ROUTE)


def _adapt_report(report: QualificationReport) -> DomainQualificationReport:
    records = tuple(_adapt_record(item) for item in report.records)
    fidelity_pass = all(
        item.evidence_fidelity.candidate_fidelity_match
        and item.evidence_fidelity.selection_fidelity_match
        and item.evidence_fidelity.adapter_fidelity_match
        and item.evidence_fidelity.invocation_fidelity_match
        and item.evidence_fidelity.output_fidelity_match
        and item.evidence_fidelity.identity_fidelity_match
        for item in report.records
        if item.case_id != "Q4-E"
    ) if report.records else False
    routing_fidelity = all(
        item.routing_decision_fidelity.routing_decision_fidelity_match
        for item in report.records
        if item.case_id != "Q4-E"
    ) if report.records else False
    authoritative = all(
        item.routing_decision_fidelity.authoritative_routing_observation_fidelity_match
        for item in report.records
        if item.case_id != "Q4-E"
    ) if report.records else False
    post_decision = all(
        not item.routing_decision_fidelity.post_decision_forcing_detected for item in report.records
    ) if report.records else True
    post_generation = all(
        not item.routing_decision_fidelity.post_generation_forcing_detected for item in report.records
    ) if report.records else True
    extra_gates = (
        QualificationGateResult(
            gate_id="routing_decision_fidelity",
            status=QualificationGateStatus.PASS if routing_fidelity else QualificationGateStatus.FAIL,
        ),
        QualificationGateResult(
            gate_id="authoritative_routing_observation_fidelity",
            status=QualificationGateStatus.PASS if authoritative else QualificationGateStatus.FAIL,
        ),
        QualificationGateResult(
            gate_id="post_decision_forcing",
            status=QualificationGateStatus.PASS if post_decision else QualificationGateStatus.FAIL,
        ),
        QualificationGateResult(
            gate_id="post_generation_forcing",
            status=QualificationGateStatus.PASS if post_generation else QualificationGateStatus.FAIL,
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
