# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Generic runner for functional qualification plugins (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from intergrax.contracts.execution_identity import mint_run_id
from intergrax.core.qualification.functional_qualification_case import QualificationNormalizedCaseResult
from intergrax.core.qualification.functional_qualification_identity import FunctionalQualificationPluginId
from intergrax.core.qualification.functional_qualification_metrics import (
    QualificationPluginMetrics,
    compute_qualification_metrics,
)
from intergrax.core.qualification.functional_qualification_plan import QualificationPlan, resolve_plan_plugins
from intergrax.core.qualification.functional_qualification_registry import QualificationPluginRegistry
from intergrax.core.qualification.functional_qualification_result import (
    QualificationPluginResult,
    QualificationRunReport,
)
from intergrax.core.qualification.functional_qualification_verdict import (
    QualificationVerdict,
    aggregate_qualification_verdicts,
)
from intergrax.knowledge.contracts.validation import JsonObject


class QualificationRunnerInfrastructureError(Exception):
    """Unexpected infrastructure failure during qualification execution."""


def _aggregate_metrics(results: tuple[QualificationPluginResult, ...]) -> QualificationPluginMetrics:
    all_cases: list[QualificationNormalizedCaseResult] = []
    for result in results:
        all_cases.extend(result.case_results)
    return compute_qualification_metrics(tuple(all_cases), repeatability_groups=())


def _verify_cross_domain_isolation(results: tuple[QualificationPluginResult, ...]) -> bool:
    seen_scopes: set[tuple[str, str, str]] = set()
    for result in results:
        for case in result.case_results:
            if case.tenant_id is None or not case.task_id or not case.run_id:
                continue
            scope = (case.tenant_id, case.task_id, case.run_id)
            if scope in seen_scopes:
                return False
            seen_scopes.add(scope)
    return True


def _verify_collision_safety(results: tuple[QualificationPluginResult, ...]) -> bool:
    scopes_by_case: dict[str, set[tuple[str, str, str]]] = {}
    for result in results:
        for case in result.case_results:
            if case.tenant_id is None or not case.task_id or not case.run_id:
                continue
            scope = (case.tenant_id, case.task_id, case.run_id)
            scopes = scopes_by_case.setdefault(case.case_id, set())
            if scope in scopes:
                return False
            scopes.add(scope)
    return True


def _analyzer_identity(results: tuple[QualificationPluginResult, ...]) -> tuple[str, str]:
    if not results:
        return ("", "")
    first = results[0]
    for result in results[1:]:
        if result.analyzer_class != first.analyzer_class or result.analyzer_module != first.analyzer_module:
            return ("MIXED", "MIXED")
    return (first.analyzer_class, first.analyzer_module)


def run_qualification_plan(
    plan: QualificationPlan,
    registry: QualificationPluginRegistry,
) -> QualificationRunReport:
    plugin_ids = resolve_plan_plugins(plan, registry)
    run_id = mint_run_id()
    plugin_results: list[QualificationPluginResult] = []

    for plugin_id in plugin_ids:
        plugin = registry.get(plugin_id)
        try:
            result = plugin.execute()
        except QualificationRunnerInfrastructureError:
            raise
        except Exception as exc:
            raise QualificationRunnerInfrastructureError(
                f"plugin_execution_failed:{plugin_id.value}:{exc}",
            ) from exc
        plugin_results.append(result)
        if (
            not plan.continue_on_plugin_failure
            and result.verdict is not QualificationVerdict.PASS
        ):
            break

    results_tuple = tuple(plugin_results)
    verdict = aggregate_qualification_verdicts(tuple(item.verdict for item in results_tuple))
    analyzer_class, analyzer_module = _analyzer_identity(results_tuple)
    return QualificationRunReport(
        schema_version="functional_qualification_report_v1",
        run_id=run_id,
        verdict=verdict,
        plan_plugin_ids=plugin_ids,
        plugin_results=results_tuple,
        aggregate_metrics=_aggregate_metrics(results_tuple),
        analyzer_identity=(analyzer_class, analyzer_module),
        domain_specific_analyzer_count=0,
        extension_change_surface=_empty_extension_surface(),
        cross_domain_isolation_pass=_verify_cross_domain_isolation(results_tuple),
        collision_safety_pass=_verify_collision_safety(results_tuple),
    )


def _empty_extension_surface() -> JsonObject:
    return {
        "core_production_changes": 0,
        "analyzer_changes": 0,
        "registry_implementation_changes": 0,
        "runner_implementation_changes": 0,
    }


__all__ = [
    "QualificationRunnerInfrastructureError",
    "run_qualification_plan",
]
