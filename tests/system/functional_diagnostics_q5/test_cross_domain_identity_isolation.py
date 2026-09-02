# © Artur Czarnecki. All rights reserved.

"""Cross-domain identity isolation tests for Q5."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_diagnostic_expectation import (
    QualificationCaseComparison,
    QualificationComparisonResult,
    QualificationFunctionalOutcome,
)
from intergrax.core.qualification.functional_qualification_case import QualificationNormalizedCaseResult
from intergrax.core.qualification.functional_qualification_fidelity import QualificationGateResult, QualificationGateStatus
from intergrax.core.qualification.functional_qualification_identity import (
    FunctionalQualificationPluginId,
    RAG_PLUGIN_ID,
    TOOL_SELECTION_PLUGIN_ID,
    WEB_SEARCH_PLUGIN_ID,
)
from intergrax.core.qualification.functional_qualification_metrics import QualificationPluginMetrics
from intergrax.core.qualification.functional_qualification_plan import QualificationPlan
from intergrax.core.qualification.functional_qualification_plugin import QualificationPluginDescriptor
from intergrax.core.qualification.functional_qualification_registry import QualificationPluginRegistry
from intergrax.core.qualification.functional_qualification_result import QualificationPluginResult
from intergrax.core.qualification.functional_qualification_runner import run_qualification_plan
from intergrax.core.qualification.functional_qualification_verdict import QualificationVerdict
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id

pytestmark = pytest.mark.unit


def _plugin_result(
    plugin_id: FunctionalQualificationPluginId,
    *,
    case_id: str,
    tenant_id: str,
) -> QualificationPluginResult:
    task_id = mint_task_id()
    run_id = mint_run_id()
    comparison = QualificationCaseComparison(case_id=case_id, result=QualificationComparisonResult.MATCH)
    case = QualificationNormalizedCaseResult(
        case_id=case_id,
        task_id=task_id,
        run_id=run_id,
        attempt_id=None,
        tenant_id=tenant_id,
        comparison=comparison,
        functional_outcome=QualificationFunctionalOutcome.PASSED,
        diag_first_failed_check=None,
        operator_outcome=None,
        repeat_group=None,
        identity_fidelity_match=True,
        is_healthy_case=True,
        expects_functional_failure=False,
        expects_inconclusive=False,
        expected_first_failed_check=None,
    )
    metrics = QualificationPluginMetrics(
        total_cases=1,
        matched_cases=1,
        mismatched_cases=0,
        false_positives=0,
        false_negatives=0,
        inconclusive_correct_cases=0,
        stage_matched_cases=1,
        stage_accuracy_percent=100.0,
        inconclusive_accuracy_percent=0.0,
        repeatability_pass=True,
        full_case_match_rate=100.0,
    )
    return QualificationPluginResult(
        plugin_id=plugin_id,
        verdict=QualificationVerdict.PASS,
        metrics=metrics,
        gate_results=(
            QualificationGateResult(gate_id="comparison_pass", status=QualificationGateStatus.PASS),
        ),
        case_results=(case,),
        artifact_ref=None,
        blocked_reason=None,
        report_sections=(),
        analyzer_class="FunctionalDiagnosticAnalyzer",
        analyzer_module="intergrax.runtime.diagnostics.functional_diagnostic_analyzer",
    )


class _IsolationPluginA:
    def __init__(self, plugin_id: FunctionalQualificationPluginId, case_id: str) -> None:
        self._plugin_id = plugin_id
        self._case_id = case_id

    @property
    def descriptor(self) -> QualificationPluginDescriptor:
        return QualificationPluginDescriptor(
            plugin_id=self._plugin_id,
            domain="isolation",
            version="1",
            display_name="Isolation",
            contract_version="v1",
            qualification_level="unit",
        )

    def execute(self) -> QualificationPluginResult:
        return _plugin_result(self._plugin_id, case_id=self._case_id, tenant_id="tenant-q5")


class _CollisionCasePlugin:
    def __init__(self, plugin_id: FunctionalQualificationPluginId) -> None:
        self._plugin_id = plugin_id

    @property
    def descriptor(self) -> QualificationPluginDescriptor:
        return QualificationPluginDescriptor(
            plugin_id=self._plugin_id,
            domain="collision",
            version="1",
            display_name="Collision",
            contract_version="v1",
            qualification_level="unit",
        )

    def execute(self) -> QualificationPluginResult:
        return _plugin_result(self._plugin_id, case_id="Q-COLLISION", tenant_id="tenant-collision")


def test_cross_domain_scopes_remain_unique() -> None:
    registry = QualificationPluginRegistry()
    registry.register(_IsolationPluginA(RAG_PLUGIN_ID, "Q1-A"))
    registry.register(_IsolationPluginA(TOOL_SELECTION_PLUGIN_ID, "Q2-A"))
    plan = QualificationPlan(
        plugin_ids=(
            RAG_PLUGIN_ID,
            TOOL_SELECTION_PLUGIN_ID,
        ),
    )
    report = run_qualification_plan(plan, registry)
    assert report.cross_domain_isolation_pass is True


def test_collision_case_ids_allowed_across_plugins() -> None:
    registry = QualificationPluginRegistry()
    registry.register(_CollisionCasePlugin(RAG_PLUGIN_ID))
    registry.register(_CollisionCasePlugin(WEB_SEARCH_PLUGIN_ID))
    plan = QualificationPlan(
        plugin_ids=(
            RAG_PLUGIN_ID,
            WEB_SEARCH_PLUGIN_ID,
        ),
    )
    report = run_qualification_plan(plan, registry)
    assert report.collision_safety_pass is True
