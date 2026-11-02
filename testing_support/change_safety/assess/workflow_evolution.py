# © Artur Czarnecki. All rights reserved.

"""W4-02 workflow evolution matrix WM-01..15 (data-driven, first-match, fail-closed)."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields

from testing_support.change_safety.contracts.enums import WorkflowEvolutionDecision
from testing_support.change_safety.contracts.verdicts import WorkflowEvolutionVerdict
from testing_support.change_safety.data.loaders import load_evol_x_data_document

_SAFE_FOR_INFLIGHT_ADOPT = frozenset(
    {WorkflowEvolutionDecision.EXPLICITLY_UNSUPPORTED, WorkflowEvolutionDecision.REJECT}
)


@dataclass(frozen=True)
class WorkflowChange:
    has_inflight: bool
    adopt_in_place: bool = False
    pin_evidence: bool = True
    fingerprint_changed: bool = True
    graph_spec_bumped: bool = False
    migration_declared: bool = False
    superseded_readable: bool = True


INPUT_NAMES: tuple[str, ...] = tuple(f.name for f in fields(WorkflowChange))


def load_workflow_matrix() -> tuple[dict[str, object], ...]:
    doc = load_evol_x_data_document("workflow_evolution_matrix.v1.json")
    rows = tuple(doc["rows"])
    if tuple(doc["inputs"]) != INPUT_NAMES:
        raise ValueError("workflow matrix inputs drifted from WorkflowChange fields")
    for row in rows:
        WorkflowEvolutionDecision(row["decision"])
        unknown = set(row["when"]) - set(INPUT_NAMES)
        if unknown:
            raise ValueError(f"{row['wm_id']}: unknown condition fields {sorted(unknown)}")
    if not rows or rows[-1]["when"] != {}:
        raise ValueError("workflow matrix must end with a fail-closed catch-all row")
    return rows


def _matches(change: WorkflowChange, when: dict[str, bool]) -> bool:
    return all(getattr(change, key) == value for key, value in when.items())


def classify_workflow_change(change: WorkflowChange) -> WorkflowEvolutionVerdict:
    for row in load_workflow_matrix():
        if _matches(change, row["when"]):  # type: ignore[arg-type]
            return WorkflowEvolutionVerdict(
                matrix_id=str(row["wm_id"]),
                decision=WorkflowEvolutionDecision(row["decision"]),
                witness=str(row["reason"]),
            )
    raise AssertionError("unreachable: matrix ends with catch-all")


def classify_inflight_definition_change(*, has_inflight: bool, migration_declared: bool) -> WorkflowEvolutionDecision:
    """Backward-compatible narrow entry point (in-place adoption attempt)."""
    return classify_workflow_change(
        WorkflowChange(has_inflight=has_inflight, adopt_in_place=has_inflight, migration_declared=migration_declared)
    ).decision


def enumerate_input_space() -> tuple[WorkflowChange, ...]:
    return tuple(
        WorkflowChange(**dict(zip(INPUT_NAMES, combo, strict=True)))
        for combo in itertools.product((False, True), repeat=len(INPUT_NAMES))
    )


def unsafe_inflight_adoptions() -> tuple[WorkflowChange, ...]:
    """Inputs where a running execution would adopt a changed definition in place with a permissive verdict."""
    bad = []
    for change in enumerate_input_space():
        if change.has_inflight and change.adopt_in_place and change.fingerprint_changed:
            if classify_workflow_change(change).decision not in _SAFE_FOR_INFLIGHT_ADOPT:
                bad.append(change)
    return tuple(bad)


def matrix_row_coverage() -> dict[str, int]:
    """wm_id -> number of inputs (of 128) resolved by that row. A row with 0 is unreachable (matrix defect)."""
    counts: dict[str, int] = {str(r["wm_id"]): 0 for r in load_workflow_matrix()}
    for change in enumerate_input_space():
        counts[classify_workflow_change(change).matrix_id] += 1
    return counts
