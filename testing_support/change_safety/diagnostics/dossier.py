# © Artur Czarnecki. All rights reserved.

"""W4-06 passive change dossier Q1-Q6 (read-only: no I/O, no runtime coupling, no authorization)."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from testing_support.change_safety.assess.extension_safety import ExtensionSafetyReport
from testing_support.change_safety.assess.isolation import IsolationReport
from testing_support.change_safety.assess.triggers import FreshnessVerdict
from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.contracts.enums import CompatibilityAtChangeResult
from testing_support.change_safety.contracts.impact import ImpactAssessmentResult
from testing_support.change_safety.contracts.selection import SelectionDecision
from testing_support.change_safety.contracts.verdicts import (
    ChangeDossier,
    DeploySafetyAssessment,
    DossierAnswer,
    WorkflowEvolutionVerdict,
)
from testing_support.change_safety.ports import ChangeSet

NOT_ASSESSED = "NOT_ASSESSED"

QUESTIONS: tuple[tuple[str, str], ...] = (
    ("Q1", "What changed?"),
    ("Q2", "What is affected (blast radius)?"),
    ("Q3", "Which tests were selected and why?"),
    ("Q4", "Is the change compatible and within extension boundaries?"),
    ("Q5", "Is the change safe to evolve / deploy?"),
    ("Q6", "Is this evidence fresh?"),
)


def _q1(change_set: ChangeSet | None) -> DossierAnswer:
    if change_set is None:
        return DossierAnswer(question_id="Q1", question=QUESTIONS[0][1], answer=NOT_ASSESSED)
    statuses = dict(change_set.statuses)
    body = "; ".join(f"{statuses.get(p, '?')} {p}" for p in sorted(change_set.paths)) or "no changed paths"
    return DossierAnswer(
        question_id="Q1",
        question=QUESTIONS[0][1],
        answer=f"{len(change_set.paths)} path(s): {body}",
        evidence_refs=(f"git:{change_set.base_sha}..{change_set.head_sha}",),
    )


def _q2(impact: ImpactAssessmentResult | None, isolation: IsolationReport | None) -> DossierAnswer:
    if impact is None:
        return DossierAnswer(question_id="Q2", question=QUESTIONS[1][1], answer=NOT_ASSESSED)
    kinds = Counter(n.kind for n in impact.affected)
    text = f"status={impact.status.value}; " + ", ".join(f"{k}={kinds[k]}" for k in sorted(kinds))
    if isolation is not None:
        text += f"; apps={list(isolation.blast_apps)}; escapes={len(isolation.escapes)}"
    return DossierAnswer(
        question_id="Q2",
        question=QUESTIONS[1][1],
        answer=text,
        evidence_refs=(f"impact:{impact.graph_digest}",),
    )


def _q3(selection: SelectionDecision | None) -> DossierAnswer:
    if selection is None:
        return DossierAnswer(question_id="Q3", question=QUESTIONS[2][1], answer=NOT_ASSESSED)
    rules = Counter(r.rule_id for r in selection.reasons)
    text = (
        f"status={selection.status.value}; selected={len(selection.selected)}; mandatory={len(selection.mandatory)}; "
        + ", ".join(f"{k}={rules[k]}" for k in sorted(rules))
    )
    return DossierAnswer(
        question_id="Q3",
        question=QUESTIONS[2][1],
        answer=text,
        evidence_refs=(f"selection:{selection.decision_id}",),
    )


def _q4(
    compat: tuple[tuple[str, CompatibilityAtChangeResult], ...] | None, extension: ExtensionSafetyReport | None
) -> DossierAnswer:
    if compat is None and extension is None:
        return DossierAnswer(question_id="Q4", question=QUESTIONS[3][1], answer=NOT_ASSESSED)
    parts: list[str] = []
    if compat is not None:
        parts.append("compat: " + (", ".join(f"{cid}={res.value}" for cid, res in sorted(compat)) or "none"))
    if extension is not None:
        parts.append(f"extension={extension.verdict.value}; class={extension.effective_class}")
    return DossierAnswer(question_id="Q4", question=QUESTIONS[3][1], answer="; ".join(parts))


def _q5(workflow: WorkflowEvolutionVerdict | None, deploy: DeploySafetyAssessment | None) -> DossierAnswer:
    if workflow is None and deploy is None:
        return DossierAnswer(question_id="Q5", question=QUESTIONS[4][1], answer=NOT_ASSESSED)
    parts: list[str] = []
    refs: list[str] = []
    if workflow is not None:
        parts.append(f"workflow={workflow.matrix_id}:{workflow.decision.value}")
        refs.append(f"matrix:{workflow.matrix_id}")
    if deploy is not None:
        reason = deploy.reason.value if deploy.reason else "-"
        parts.append(f"deploy={deploy.matrix_id}:{deploy.verdict.value}({reason})")
        refs.append(f"matrix:{deploy.matrix_id}")
    return DossierAnswer(question_id="Q5", question=QUESTIONS[4][1], answer="; ".join(parts), evidence_refs=tuple(refs))


def _q6(freshness: FreshnessVerdict | None) -> DossierAnswer:
    if freshness is None:
        return DossierAnswer(question_id="Q6", question=QUESTIONS[5][1], answer=NOT_ASSESSED)
    text = "FRESH" if freshness.allowed else f"STALE: {','.join(freshness.stale_triggers)}"
    return DossierAnswer(
        question_id="Q6",
        question=QUESTIONS[5][1],
        answer=f"{text} ({freshness.stale_count}/{freshness.evaluated} triggers fired)",
    )


def build_change_dossier(
    *,
    graph_digest: str,
    inventory_digest: str,
    stale: bool = False,
    change_set: ChangeSet | None = None,
    impact: ImpactAssessmentResult | None = None,
    isolation: IsolationReport | None = None,
    selection: SelectionDecision | None = None,
    compat: tuple[tuple[str, CompatibilityAtChangeResult], ...] | None = None,
    extension: ExtensionSafetyReport | None = None,
    workflow: WorkflowEvolutionVerdict | None = None,
    deploy: DeploySafetyAssessment | None = None,
    freshness: FreshnessVerdict | None = None,
    created_at_utc: str | None = None,
) -> ChangeDossier:
    """Deterministic for identical inputs: ``content_digest`` excludes ``created_at_utc``."""
    sections = (
        _q1(change_set),
        _q2(impact, isolation),
        _q3(selection),
        _q4(compat, extension),
        _q5(workflow, deploy),
        _q6(freshness),
    )
    is_stale = stale or (freshness is not None and not freshness.allowed)
    stale_triggers = freshness.stale_triggers if freshness is not None else ()
    refs = tuple(sorted({ref for s in sections for ref in s.evidence_refs}))
    digest = stable_digest_hex(
        {
            "graph": graph_digest,
            "inventory": inventory_digest,
            "stale": is_stale,
            "sections": [s.model_dump(mode="json") for s in sections],
        }
    )
    return ChangeDossier(
        created_at_utc=created_at_utc or datetime.now(UTC).replace(microsecond=0).isoformat(),
        graph_digest=graph_digest,
        inventory_digest=inventory_digest,
        stale=is_stale,
        stale_triggers=stale_triggers,
        sections=sections,
        evidence_refs=refs,
        content_digest=digest,
    )
