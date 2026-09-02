# © Artur Czarnecki. All rights reserved.

"""Extension proof: synthetic fifth plugin through generic runner."""

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

_SYNTHETIC_PLUGIN_ID = FunctionalQualificationPluginId("functional.synthetic_test")


class SyntheticQualificationPlugin:
    @property
    def descriptor(self) -> QualificationPluginDescriptor:
        return QualificationPluginDescriptor(
            plugin_id=_SYNTHETIC_PLUGIN_ID,
            domain="synthetic",
            version="1.0.0",
            display_name="Synthetic Extension Plugin",
            contract_version="functional_qualification_v1",
            qualification_level="unit",
        )

    def execute(self) -> QualificationPluginResult:
        task_id = mint_task_id()
        run_id = mint_run_id()
        comparison = QualificationCaseComparison(
            case_id="SYN-1",
            result=QualificationComparisonResult.MATCH,
        )
        case = QualificationNormalizedCaseResult(
            case_id="SYN-1",
            task_id=task_id,
            run_id=run_id,
            attempt_id=None,
            tenant_id="tenant-synthetic",
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
            plugin_id=_SYNTHETIC_PLUGIN_ID,
            verdict=QualificationVerdict.PASS,
            metrics=metrics,
            gate_results=(
                QualificationGateResult(
                    gate_id="synthetic_gate",
                    status=QualificationGateStatus.PASS,
                ),
            ),
            case_results=(case,),
            artifact_ref=None,
            blocked_reason=None,
            report_sections=(),
            analyzer_class="FunctionalDiagnosticAnalyzer",
            analyzer_module="intergrax.runtime.diagnostics.functional_diagnostic_analyzer",
        )


def test_synthetic_plugin_runs_without_core_changes() -> None:
    registry = QualificationPluginRegistry()
    registry.register(SyntheticQualificationPlugin())
    plan = QualificationPlan(plugin_ids=(_SYNTHETIC_PLUGIN_ID,))
    report = run_qualification_plan(plan, registry)
    assert report.verdict is QualificationVerdict.PASS
    assert len(report.plugin_results) == 1
    assert report.plugin_results[0].plugin_id == _SYNTHETIC_PLUGIN_ID


def test_plan_with_subset_of_plugins() -> None:
    registry = QualificationPluginRegistry()
    registry.register(SyntheticQualificationPlugin())
    plan = QualificationPlan(plugin_ids=(_SYNTHETIC_PLUGIN_ID,))
    report = run_qualification_plan(plan, registry)
    assert report.plan_plugin_ids == (_SYNTHETIC_PLUGIN_ID,)
