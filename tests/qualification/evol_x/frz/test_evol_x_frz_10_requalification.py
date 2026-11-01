# © Artur Czarnecki. All rights reserved.

"""FRZ-EVOL-10: post-freeze requalification triggers and evidence freshness enforced (T1..T9)."""

import pytest
from testing_support.change_safety.assess.triggers import (
    build_baseline_manifest,
    compute_actual_digests,
    evaluate_freshness,
    trigger_table,
)
from tests.qualification.evol_x._constants import FRZ_EVOL_10_SCOPE_FOOTNOTE

pytestmark = [pytest.mark.qualification, pytest.mark.unit]


def test_frz_evol_10_each_trigger_fires_independently_on_real_digests(evol_inventory, evol_head) -> None:  # noqa: ANN001
    actual = compute_actual_digests(evol_inventory)
    baseline = build_baseline_manifest(accepted_at_sha=evol_head, digests=actual)
    fresh = evaluate_freshness(baseline, actual)
    assert fresh.allowed and fresh.evaluated == 9 and fresh.stale_count == 0
    for row in trigger_table():
        verdict = evaluate_freshness(baseline, dict(actual, **{row["digest_key"]: "tampered"}))
        assert verdict.stale_triggers == (row["trigger_id"],) and not verdict.allowed


def test_frz_evol_10_missing_baseline_digest_is_stale_never_fresh(evol_inventory, evol_head) -> None:  # noqa: ANN001
    actual = compute_actual_digests(evol_inventory)
    baseline = build_baseline_manifest(accepted_at_sha=evol_head, digests={})
    assert evaluate_freshness(baseline, actual).stale_count == 9
    assert "T1..T9" in FRZ_EVOL_10_SCOPE_FOOTNOTE
