# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-08: passive change dossier answers Q1-Q6 (what changed / what failed / which change introduced)."""

import pytest
from testing_support.change_safety.assess.isolation import assess_isolation
from testing_support.change_safety.assess.triggers import (
    build_baseline_manifest,
    compute_actual_digests,
    evaluate_freshness,
)
from testing_support.change_safety.diagnostics.dossier import NOT_ASSESSED, QUESTIONS, build_change_dossier
from testing_support.change_safety.graph.impact import assess_path_change_impact
from testing_support.change_safety.ports import ChangeSet
from tests.qualification.evol_x._constants import FRZ_EVOL_08_SCOPE_FOOTNOTE

pytestmark = [pytest.mark.qualification, pytest.mark.unit]

_PATH = "intergrax/contracts/execution_identity.py"


def test_frz_evol_08_dossier_answers_all_questions_on_real_change(evol_inventory, evol_head) -> None:  # noqa: ANN001
    changed = frozenset({_PATH})
    impact = assess_path_change_impact(evol_inventory, changed_paths=changed)
    actual = compute_actual_digests(evol_inventory)
    baseline = build_baseline_manifest(accepted_at_sha=evol_head, digests=actual)
    dossier = build_change_dossier(
        graph_digest=impact.graph_digest,
        inventory_digest=evol_inventory.inventory_digest,
        change_set=ChangeSet(paths=changed, base_sha=evol_head, head_sha=evol_head, statuses=((_PATH, "M"),)),
        impact=impact,
        isolation=assess_isolation(impact, changed_paths=changed),
        freshness=evaluate_freshness(baseline, actual),
    )
    answered = {s.question_id: s.answer for s in dossier.sections}
    assert list(answered) == [q for q, _ in QUESTIONS]
    for qid in ("Q1", "Q2", "Q6"):
        assert answered[qid] != NOT_ASSESSED
    assert not dossier.stale and dossier.content_digest


def test_frz_evol_08_stale_evidence_is_flagged_and_scope_is_passive(evol_inventory, evol_head) -> None:  # noqa: ANN001
    actual = compute_actual_digests(evol_inventory)
    baseline = build_baseline_manifest(accepted_at_sha=evol_head, digests=actual)
    stale = evaluate_freshness(baseline, dict(actual, inventory_digest="0" * 64))
    dossier = build_change_dossier(graph_digest="g", inventory_digest="i", freshness=stale)
    assert dossier.stale and dossier.stale_triggers == ("T2",)
    assert "passive" in FRZ_EVOL_08_SCOPE_FOOTNOTE
