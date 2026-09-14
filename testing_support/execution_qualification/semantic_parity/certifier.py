# © Artur Czarnecki. All rights reserved.

"""Reusable global semantic parity certifier for canonical qualification profiles."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from testing_support.execution_qualification.catalog.catalog import QualificationCatalog
from testing_support.execution_qualification.catalog.expansion import (
    unique_required_leaf_targets,
)
from testing_support.execution_qualification.contracts import (
    QualificationRunConfig,
    QualificationRunStatus,
    QualificationSuiteStatus,
)
from testing_support.execution_qualification.coordinator import QualificationCoordinator
from testing_support.execution_qualification.fake_executor import (
    FakeQualificationSuiteExecutor,
)
from testing_support.execution_qualification.coordinator_port import (
    QualificationCoordinatorPort,
)
from testing_support.execution_qualification.graph_contracts import (
    QualificationExecutionPlan,
    QualificationNodeKind,
    QualificationPlanRunResult,
)
from testing_support.execution_qualification.plan_runner import (
    run_qualification_execution_plan,
)
from testing_support.execution_qualification.semantic_parity.graph_analysis import (
    canonical_leaf_pytest_argument_set,
    nodes_reachable_from_roots,
    reachable_non_leaf_gate_count,
    root_ids_transitively_requiring_leaf,
)
from testing_support.execution_qualification.semantic_parity.models import (
    CoverageParityDiagnostics,
    LeafInjectionParityRow,
    QualificationProfileParityResult,
    QualificationSemanticParityCase,
    QualificationSemanticParityReport,
    SemanticParityCertificationStatus,
)

SuiteExecutorFactory = Callable[
    [Mapping[str, QualificationSuiteStatus]],
    QualificationCoordinatorPort,
]


def _legacy_leaf_argument_set(
    case: QualificationSemanticParityCase,
) -> frozenset[tuple[str, ...]]:
    return frozenset(
        entry.pytest_arguments
        for entry in unique_required_leaf_targets(case.legacy_semantic_source)
    )


def _default_executor_factory(
    overrides: Mapping[str, QualificationSuiteStatus],
) -> QualificationCoordinatorPort:
    fake = FakeQualificationSuiteExecutor({}, status_overrides=overrides)
    return QualificationCoordinator(executor=fake)


class QualificationSemanticParityCertifier:
    def __init__(
        self,
        catalog: QualificationCatalog,
        *,
        repo_root: Path,
        run_artifact_root: Path,
        executor_factory: SuiteExecutorFactory | None = None,
    ) -> None:
        self._catalog = catalog
        self._repo_root = repo_root
        self._run_artifact_root = run_artifact_root
        self._executor_factory = (
            executor_factory
            if executor_factory is not None
            else _default_executor_factory
        )

    def certify_all(
        self,
        cases: tuple[QualificationSemanticParityCase, ...],
    ) -> QualificationSemanticParityReport:
        results = tuple(self.certify_profile(case) for case in cases)
        overall = SemanticParityCertificationStatus.PASS
        if any(not result.profile_pass for result in results):
            overall = SemanticParityCertificationStatus.BLOCKED
        return QualificationSemanticParityReport(
            profile_results=results,
            overall_status=overall,
        )

    def certify_profile(
        self,
        case: QualificationSemanticParityCase,
    ) -> QualificationProfileParityResult:
        failures: list[str] = []
        compiled_first = self._catalog.compile_profile(case.canonical_profile_id)
        compiled_second = self._catalog.compile_profile(case.canonical_profile_id)
        plan = compiled_first.plan
        determinism_parity = compiled_first.plan == compiled_second.plan
        if not determinism_parity:
            failures.append("compile plan not deterministic")

        if plan.root_gate_ids != case.expected_root_ids:
            failures.append(
                f"root_gate_ids mismatch: expected {case.expected_root_ids!r}, "
                f"got {plan.root_gate_ids!r}",
            )

        legacy_set = _legacy_leaf_argument_set(case)
        canonical_set = canonical_leaf_pytest_argument_set(plan)
        missing = legacy_set - canonical_set
        extra = canonical_set - legacy_set
        coverage_parity = not missing and not extra
        coverage_diag = CoverageParityDiagnostics(
            missing_in_canonical=frozenset(missing),
            unexpected_in_canonical=frozenset(extra),
        )
        if missing:
            failures.append(f"missing_in_canonical={sorted(missing)!r}")
        if extra:
            failures.append(f"unexpected_in_canonical={sorted(extra)!r}")

        reachable = nodes_reachable_from_roots(plan)
        leaf_set = set(plan.leaf_suite_ids)
        unreachable_leaves = leaf_set - set(reachable)
        reachability_parity = not unreachable_leaves
        if unreachable_leaves:
            failures.append(f"unreachable leaves: {sorted(unreachable_leaves)!r}")

        gate_count = sum(
            1
            for node in plan.ordered_nodes
            if node.kind is not QualificationNodeKind.LEAF_SUITE
        )

        all_pass = self._run_plan(plan, {})
        all_pass_parity = all_pass.status is QualificationRunStatus.PASS
        if not all_pass_parity:
            failures.append("all-PASS run did not PASS")

        failure_rows, failure_ok = self._injection_matrix(
            case.profile_id,
            plan,
            QualificationSuiteStatus.FAIL,
        )
        if not failure_ok:
            failures.append("FAIL injection matrix mismatch")

        skip_rows, skip_ok = self._injection_matrix(
            case.profile_id,
            plan,
            QualificationSuiteStatus.SKIP,
        )
        if not skip_ok:
            failures.append("SKIP injection matrix mismatch")

        collect_all_parity = self._verify_collect_all(plan)
        if not collect_all_parity:
            failures.append("collect-all violated under single-leaf FAIL")

        receipt_parity = self._verify_receipt_identity(plan, all_pass)
        if not receipt_parity:
            failures.append("root or suite receipt identity mismatch")

        gate_semantics_parity = self._verify_gate_receipts(plan, all_pass)
        if not gate_semantics_parity:
            failures.append("gate receipt semantics mismatch")

        physical_dedup_parity = self._verify_physical_dedup(plan)
        if not physical_dedup_parity:
            failures.append("physical dedup violated")

        execution_deterministic = self._verify_execution_determinism(plan)
        if not execution_deterministic:
            failures.append("injected execution not deterministic")

        if all_pass.gate_count != gate_count:
            failures.append(
                f"gate_count {all_pass.gate_count} != expected {gate_count}",
            )

        if all_pass.physical_leaf_count != len(plan.leaf_suite_ids):
            failures.append("physical_leaf_count != len(leaf_suite_ids)")

        return QualificationProfileParityResult(
            profile_id=case.profile_id,
            coverage_parity=coverage_parity,
            all_pass_parity=all_pass_parity,
            failure_injection_parity=failure_ok,
            skip_injection_parity=skip_ok,
            collect_all_parity=collect_all_parity,
            receipt_parity=receipt_parity,
            reachability_parity=reachability_parity,
            determinism_parity=determinism_parity,
            gate_semantics_parity=gate_semantics_parity,
            physical_dedup_parity=physical_dedup_parity,
            deterministic=execution_deterministic,
            leaf_count=len(plan.leaf_suite_ids),
            gate_count=gate_count,
            coverage_diagnostics=coverage_diag if not coverage_parity else None,
            failure_rows=failure_rows,
            skip_rows=skip_rows,
            invariant_failures=tuple(failures),
        )

    def _run_config(self, run_id: str) -> QualificationRunConfig:
        return QualificationRunConfig(
            repo_root=self._repo_root,
            max_parallel=4,
            run_artifact_root=self._run_artifact_root / run_id,
            suite_timeout_seconds=30.0,
            run_id=run_id,
        )

    def _run_plan(
        self,
        plan: QualificationExecutionPlan,
        overrides: Mapping[str, QualificationSuiteStatus],
    ) -> QualificationPlanRunResult:
        coordinator = self._executor_factory(overrides)
        return run_qualification_execution_plan(
            plan,
            self._run_config(f"parity-{plan.profile_id}-{len(overrides)}"),
            coordinator=coordinator,
        )

    def _injection_matrix(
        self,
        profile_id: str,
        plan: QualificationExecutionPlan,
        injected_status: QualificationSuiteStatus,
    ) -> tuple[tuple[LeafInjectionParityRow, ...], bool]:
        rows: list[LeafInjectionParityRow] = []
        ok = True
        for suite_id in plan.leaf_suite_ids:
            affected = root_ids_transitively_requiring_leaf(plan, suite_id)
            overrides = {suite_id: injected_status}
            result = self._run_plan(plan, overrides)
            actual_roots = tuple(
                (receipt.gate_id, receipt.status)
                for receipt in result.root_gate_receipts
            )
            for root_id in affected:
                root_receipt = next(
                    r for r in result.root_gate_receipts if r.gate_id == root_id
                )
                if root_receipt.status is not QualificationSuiteStatus.FAIL:
                    ok = False
            if injected_status is QualificationSuiteStatus.SKIP:
                if result.status is QualificationRunStatus.PASS:
                    ok = False
            rows.append(
                LeafInjectionParityRow(
                    profile_id=profile_id,
                    suite_id=suite_id,
                    affected_root_ids=affected,
                    expected_root_status=QualificationSuiteStatus.FAIL,
                    actual_root_status=actual_roots,
                ),
            )
        return tuple(rows), ok

    def _verify_collect_all(self, plan: QualificationExecutionPlan) -> bool:
        if not plan.leaf_suite_ids:
            return True
        failing_leaf = plan.leaf_suite_ids[0]
        result = self._run_plan(
            plan,
            {failing_leaf: QualificationSuiteStatus.FAIL},
        )
        for suite_id in plan.leaf_suite_ids:
            if suite_id == failing_leaf:
                continue
            receipt = next(r for r in result.suite_receipts if r.suite_id == suite_id)
            if receipt.status is not QualificationSuiteStatus.PASS:
                return False
        return True

    def _verify_receipt_identity(
        self,
        plan: QualificationExecutionPlan,
        result: QualificationPlanRunResult,
    ) -> bool:
        if tuple(r.gate_id for r in result.root_gate_receipts) != plan.root_gate_ids:
            return False
        if len(result.suite_receipts) != len(plan.leaf_suite_ids):
            return False
        for receipt, expected_id in zip(
            result.suite_receipts,
            plan.leaf_suite_ids,
            strict=True,
        ):
            if receipt.suite_id != expected_id:
                return False
        return True

    def _verify_gate_receipts(
        self,
        plan: QualificationExecutionPlan,
        result: QualificationPlanRunResult,
    ) -> bool:
        node_by_id = {node.node_id: node for node in plan.ordered_nodes}
        gate_ids_seen: set[str] = set()
        for receipt in result.gate_receipts:
            if receipt.gate_id in gate_ids_seen:
                return False
            gate_ids_seen.add(receipt.gate_id)
            node = node_by_id.get(receipt.gate_id)
            if node is None or node.gate is None:
                return False
            if receipt.consumed_node_ids != node.dependencies:
                return False
            if receipt.status is QualificationSuiteStatus.PASS:
                if receipt.failure_dependencies:
                    return False
            else:
                if not receipt.failure_dependencies:
                    return False
        return len(gate_ids_seen) == reachable_non_leaf_gate_count(plan)

    def _verify_physical_dedup(self, plan: QualificationExecutionPlan) -> bool:
        fake = FakeQualificationSuiteExecutor({})
        coordinator = QualificationCoordinator(executor=fake)
        run_qualification_execution_plan(
            plan,
            self._run_config(f"dedup-{plan.profile_id}"),
            coordinator=coordinator,
        )
        for suite_id in plan.leaf_suite_ids:
            if fake.invocation_counts.get(suite_id, 0) != 1:
                return False
        return True

    def _verify_execution_determinism(self, plan: QualificationExecutionPlan) -> bool:
        if not plan.leaf_suite_ids:
            return True
        target = plan.leaf_suite_ids[-1]
        overrides = {target: QualificationSuiteStatus.FAIL}
        first = self._run_plan(plan, overrides)
        second = self._run_plan(plan, overrides)
        return (
            first.status == second.status
            and first.root_gate_receipts == second.root_gate_receipts
            and tuple(r.status for r in first.suite_receipts)
            == tuple(r.status for r in second.suite_receipts)
        )
