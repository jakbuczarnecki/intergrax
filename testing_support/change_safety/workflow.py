# © Artur Czarnecki. All rights reserved.

"""W4-08 EVOL-X change workflow pipeline A1..A8 (qualification tooling; passive, fail-closed)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from testing_support.change_safety.assess.consumers import count_critical_consumer_omissions
from testing_support.change_safety.assess.deployment_safety import DeployFacts, assess_deploy_safety
from testing_support.change_safety.assess.extension_safety import ExtensionVerdict, assess_extension_safety
from testing_support.change_safety.assess.isolation import assess_isolation
from testing_support.change_safety.assess.triggers import compute_actual_digests, evaluate_freshness
from testing_support.change_safety.assess.workflow_evolution import WorkflowChange, classify_workflow_change
from testing_support.change_safety.contracts.enums import (
    CompatibilityAtChangeResult,
    DeploySafetyVerdict,
    ImpactStatus,
    SelectionStatus,
    WorkflowEvolutionDecision,
)
from testing_support.change_safety.contracts.verdicts import AcceptedBaselineManifest
from testing_support.change_safety.diagnostics.dossier import build_change_dossier
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.closed_world_gates import (
    run_cw01,
    run_cw02,
    run_cw03,
    run_cw04,
    run_cw06,
)
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.ports import ChangeSet
from testing_support.change_safety.selection.selector import SelectionInputs, select_tests

PASS = "PASS"
FAIL = "FAIL"
SKIPPED = "SKIPPED"
EXIT_OK = 0
EXIT_STAGE_FAILED = 2

STAGE_NAMES = (
    ("A1", "intake"),
    ("A2", "inventory_gates"),
    ("A3", "impact"),
    ("A4", "compat"),
    ("A5", "selection"),
    ("A6", "safety"),
    ("A7", "dossier"),
    ("A8", "freshness"),
)


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    name: str
    status: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowRunReport:
    stages: tuple[StageResult, ...]

    @property
    def failed_stages(self) -> tuple[str, ...]:
        return tuple(s.stage_id for s in self.stages if s.status == FAIL)

    @property
    def exit_code(self) -> int:
        return EXIT_STAGE_FAILED if self.failed_stages else EXIT_OK

    def stage(self, stage_id: str) -> StageResult:
        return next(s for s in self.stages if s.stage_id == stage_id)


def run_change_workflow(
    *,
    change_set: ChangeSet,
    inventory: ClosedWorldInventory,
    selection_inputs_factory: Callable[[Any], SelectionInputs],
    owner_rules: dict[str, Any],
    exclusion_register: dict[str, Any],
    baseline: AcceptedBaselineManifest | None,
    compat: tuple[tuple[str, CompatibilityAtChangeResult], ...] | None = None,
    workflow_change: WorkflowChange | None = None,
    deploy_facts: DeployFacts | None = None,
    authorized_surfaces: frozenset[str] = frozenset(),
    declared_shared: frozenset[str] = frozenset(),
) -> WorkflowRunReport:
    stages: list[StageResult] = []

    # A1 intake
    stages.append(StageResult("A1", "intake", PASS, f"{len(change_set.paths)} changed path(s)"))

    # A2 inventory gates (cheap, deterministic subset CW-01..04, CW-06)
    gates = (
        run_cw01(inventory, exclusion_register),
        run_cw02(inventory, owner_rules),
        run_cw03(inventory),
        run_cw04(inventory, exclusion_register),
        run_cw06(inventory),
    )
    bad = [g.gate_id for g in gates if not g.passed]
    stages.append(
        StageResult(
            "A2", "inventory_gates", FAIL if bad else PASS, f"failed gates: {bad}" if bad else "CW-01..04,06 ok"
        )
    )

    # A3 impact
    impact = assess_path_change_impact(inventory, changed_paths=change_set.paths)
    impact_bad = impact.status in {ImpactStatus.INTEGRITY_FAILURE, ImpactStatus.REFUSED}
    stages.append(
        StageResult(
            "A3",
            "impact",
            FAIL if impact_bad else PASS,
            f"status={impact.status.value}; affected={len(impact.affected)}",
        )
    )

    # A4 compat (explicit contract verdicts supplied by the operator/adapter)
    if compat is None:
        stages.append(StageResult("A4", "compat", SKIPPED, "no contract changes supplied"))
    else:
        flagged = sorted(
            cid
            for cid, res in compat
            if res in {CompatibilityAtChangeResult.INCOMPATIBLE, CompatibilityAtChangeResult.UNKNOWN}
        )
        stages.append(
            StageResult(
                "A4", "compat", FAIL if flagged else PASS, f"incompatible/unknown: {flagged}" if flagged else "ok"
            )
        )

    # A5 selection (fail-closed)
    selection = select_tests(
        replace(
            selection_inputs_factory(impact),
            critical_omissions=count_critical_consumer_omissions(inventory),
            expected_graph_digest=impact.graph_digest,
            expected_inventory_digest=inventory.inventory_digest,
        )
    )
    sel_bad = selection.status is SelectionStatus.REFUSED_INTEGRITY
    stages.append(
        StageResult(
            "A5",
            "selection",
            FAIL if sel_bad else PASS,
            f"status={selection.status.value}; selected={len(selection.selected)}",
        )
    )

    # A6 safety assessments
    isolation = assess_isolation(impact, changed_paths=change_set.paths, declared_shared=declared_shared)
    extension = assess_extension_safety(changed_paths=change_set.paths, authorized_surfaces=authorized_surfaces)
    workflow = classify_workflow_change(workflow_change) if workflow_change is not None else None
    deploy = assess_deploy_safety(deploy_facts) if deploy_facts is not None else None
    problems: list[str] = []
    if not isolation.isolated:
        problems.append(f"isolation escapes={len(isolation.escapes)}")
    if extension.verdict is ExtensionVerdict.CORE_TOUCH_UNAUTHORIZED:
        problems.append(f"unauthorized core surfaces={list(extension.unauthorized_surfaces)}")
    if workflow is not None and workflow.decision in {
        WorkflowEvolutionDecision.REJECT,
        WorkflowEvolutionDecision.EXPLICITLY_UNSUPPORTED,
    }:
        problems.append(f"workflow {workflow.matrix_id}={workflow.decision.value}")
    if deploy is not None and deploy.verdict is DeploySafetyVerdict.REFUSE:
        problems.append(f"deploy {deploy.matrix_id}=REFUSE")
    stages.append(StageResult("A6", "safety", FAIL if problems else PASS, "; ".join(problems) or "ok"))

    # A8 freshness first (A7 dossier embeds it)
    actual = compute_actual_digests(inventory)
    freshness = evaluate_freshness(baseline, actual) if baseline is not None else None

    # A7 dossier
    dossier = build_change_dossier(
        graph_digest=impact.graph_digest,
        inventory_digest=inventory.inventory_digest,
        change_set=change_set,
        impact=impact,
        isolation=isolation,
        selection=selection,
        compat=compat,
        extension=extension,
        workflow=workflow,
        deploy=deploy,
        freshness=freshness,
    )
    stages.append(StageResult("A7", "dossier", PASS, f"digest={dossier.content_digest[:16]}"))

    # A8 freshness verdict (no baseline => cannot prove freshness => FAIL, never silent pass)
    if freshness is None:
        stages.append(StageResult("A8", "freshness", FAIL, "no accepted baseline manifest supplied"))
    else:
        stages.append(
            StageResult(
                "A8",
                "freshness",
                PASS if freshness.allowed else FAIL,
                "fresh" if freshness.allowed else f"stale triggers: {list(freshness.stale_triggers)}",
            )
        )

    ordered = sorted(stages, key=lambda s: s.stage_id)
    payloads = {
        "A3": impact.model_dump(mode="json"),
        "A5": selection.model_dump(mode="json"),
        "A7": dossier.model_dump(mode="json"),
    }
    return WorkflowRunReport(
        stages=tuple(StageResult(s.stage_id, s.name, s.status, s.detail, payloads.get(s.stage_id, {})) for s in ordered)
    )
