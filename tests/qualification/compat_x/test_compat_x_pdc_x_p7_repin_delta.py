# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-L: the re-pinned COMPAT-X counters are bound to an exact, attributed id delta (debt item Z-D1).

The pinned counters of ``_compat_x_semantic_identity_reconciliation`` moved with ticketed product changes after the
replay lineage ``05549f9``. These tests prove the arithmetic of every pin against the id-exact evidence fixture
``tests/fixtures/compat_x/repin_pdc_x_p7_delta.json`` (counts never stand alone), that every added id exists and every
removed id is gone, that the four new shim paths are heuristic hits, and that the gates still fail on a hidden removal,
an extra candidate, a balanced substitution and an unexpected or missing shim path.
"""

from __future__ import annotations

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
from tests.qualification.compat_x import _compat_x_semantic_identity_reconciliation as recon
from tests.qualification.compat_x._compat_x_classifiers import classify_shim_module
from tests.qualification.compat_x._compat_x_closed_world import (
    ClosedWorldReport,
    _shim_candidates_for_module,
    build_closed_world_report,
)
from tests.qualification.compat_x._compat_x_deprecation_r5_shim_reconciliation import (
    load_replay_x_accepted_shim_delta,
    shim_delta_accepted_r5_gate_violations,
)
from tests.qualification.compat_x._compat_x_semantic_identity_reconciliation import (
    ACCEPTED_CURRENT_HEAD_EXCLUSIONS,
    ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES,
    ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES,
    ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES,
    PDC_X_P7_REPIN_MEASURED_AT_SHA,
    REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256,
    REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES,
    REPLAY_X_COMPAT_LINEAGE_HEAD_SHA,
    _closed_world_report_with_substituted_raw_candidates,
    _remainder_stable_inventory_fingerprint,
    _synthetic_r3_inventory_probe_candidate,
    build_semantic_identity_reconciliation_closure,
    stable_inventory_candidate_identity,
    validate_current_head_compat_closed_world_inventory,
)

pytestmark = pytest.mark.unit

_FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "compat_x" / "repin_pdc_x_p7_delta.json"
_KNOWN_GROUPS = frozenset({"D1", "D4", "D5", "D6", "D7", "D8", "MCP-SERVER-X"})


@lru_cache(maxsize=1)
def _fx() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _current_identities() -> dict[str, str]:
    return {
        stable_inventory_candidate_identity(c): c.discovery_kind for c in build_closed_world_report().raw_candidates
    }


def _report_without(identity: str) -> ClosedWorldReport:
    raw = tuple(
        c for c in build_closed_world_report().raw_candidates if stable_inventory_candidate_identity(c) != identity
    )
    assert len(raw) == len(build_closed_world_report().raw_candidates) - 1, identity
    return _closed_world_report_with_substituted_raw_candidates(raw)


# ------------------------------------------------------------------ the pinned constants are bound to the id delta


def test_the_fixture_names_the_measurement_and_the_baseline() -> None:
    fx = _fx()
    assert fx["ticket"] == "T-IGX-041-L"
    assert fx["baseline_lineage_head_sha"] == REPLAY_X_COMPAT_LINEAGE_HEAD_SHA
    assert fx["measured_at_sha"] == PDC_X_P7_REPIN_MEASURED_AT_SHA
    assert len(PDC_X_P7_REPIN_MEASURED_AT_SHA) == 40


def test_the_raw_and_remainder_pins_equal_previous_plus_added_minus_removed() -> None:
    fx = _fx()
    added, removed = len(fx["added_candidate_ids"]), len(fx["removed_candidate_ids"])
    assert fx["previous"]["raw_candidates"] + added - removed == ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES
    assert fx["previous"]["remainder_raw_candidates"] + added - removed == REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES
    assert fx["repinned"]["raw_candidates"] == ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES
    assert fx["repinned"]["remainder_raw_candidates"] == REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES
    assert fx["repinned"]["remainder_fingerprint_sha256"] == REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256
    assert fx["previous"]["remainder_fingerprint_sha256"] != REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256


def test_the_surface_and_exclusion_pins_equal_previous_plus_added_minus_removed() -> None:
    fx = _fx()
    surfaces = fx["previous"]["semantic_surfaces"] + len(fx["added_surface_ids"]) - len(fx["removed_surface_ids"])
    exclusions = fx["previous"]["exclusions"] + len(fx["added_exclusion_ids"]) - len(fx["removed_exclusion_ids"])
    assert surfaces == ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES == fx["repinned"]["semantic_surfaces"]
    assert exclusions == ACCEPTED_CURRENT_HEAD_EXCLUSIONS == fx["repinned"]["exclusions"]
    assert (
        ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES == fx["repinned"]["legacy_projected_semantic_surfaces"]
    )
    assert (
        fx["previous"]["legacy_projected_semantic_surfaces"] < ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES
    )


def test_every_candidate_of_the_delta_is_attributed_to_a_ticketed_change() -> None:
    fx = _fx()
    for row in fx["added_candidate_ids"] + fx["removed_candidate_ids"]:
        assert row["group"] in _KNOWN_GROUPS, row
        assert row["owning_change"] == fx["groups"][row["group"]], row
        assert row["kind"], row
    ids = [row["id"] for row in fx["added_candidate_ids"]]
    assert len(ids) == len(set(ids))
    assert not {row["id"] for row in fx["added_candidate_ids"]} & {row["id"] for row in fx["removed_candidate_ids"]}


# ------------------------------------------------------------------ the id delta is true of the current inventory


def test_every_added_candidate_is_in_the_current_inventory_with_its_recorded_kind() -> None:
    current = _current_identities()
    for row in _fx()["added_candidate_ids"]:
        assert current.get(row["id"]) == row["kind"], row


def test_no_removed_candidate_is_in_the_current_inventory() -> None:
    current = _current_identities()
    for row in _fx()["removed_candidate_ids"]:
        assert row["id"] not in current, row


def test_every_added_surface_and_exclusion_is_present_and_every_removed_one_is_gone() -> None:
    fx = _fx()
    report = build_closed_world_report()
    surfaces = {s.surface_id for s in report.semantic_surfaces}
    exclusions = {(e.exclusion_id, e.reason) for e in report.exclusions}
    assert set(fx["added_surface_ids"]) <= surfaces
    assert not set(fx["removed_surface_ids"]) & surfaces
    assert {(row["id"], row["reason"]) for row in fx["added_exclusion_ids"]} <= exclusions
    assert not {(row["id"], row["reason"]) for row in fx["removed_exclusion_ids"]} & exclusions


def test_the_legacy_projected_pin_matches_the_reconciliation_closure() -> None:
    closure = build_semantic_identity_reconciliation_closure(build_closed_world_report())
    assert closure.legacy_projected_semantic_count == ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES


# ------------------------------------------------------------------ the shim delta is classified


def test_the_accepted_shim_delta_is_the_previous_paths_plus_the_four_heuristic_hits() -> None:
    fx = _fx()
    hits = tuple(h["path"] for h in fx["shim_heuristic_hits"])
    assert len(hits) == len(set(hits)) == 4
    previous = tuple(fx["previous"]["shim_delta_added_paths"])
    assert len(previous) == 25
    assert not set(hits) & set(previous)
    accepted = load_replay_x_accepted_shim_delta()
    assert accepted.added_paths == tuple(sorted((*previous, *hits)))
    assert accepted.added_paths == tuple(sorted(fx["repinned"]["shim_delta_added_paths"]))
    assert accepted.removed_paths == ("intergrax/runtime/execution/inference_composition.py",)


@pytest.mark.parametrize("hit_index", range(4))
def test_each_new_shim_path_reproduces_as_a_heuristic_hit_not_a_compatibility_shim(hit_index: int) -> None:
    hit = _fx()["shim_heuristic_hits"][hit_index]
    path = Path(__file__).resolve().parents[3] / hit["path"]
    source = path.read_text(encoding="utf-8")
    candidates = _shim_candidates_for_module(hit["path"], source)
    assert [c.discovered_signal for c in candidates] == [hit["signal"]]
    assert hit["signal"] == "source.contains:adapter"
    assert classify_shim_module(hit["path"], source).value == hit["class"] == "TRANSLATION_ONLY"
    assert hit["classification"] == "heuristic hit, not a compatibility shim"
    line = source.splitlines()[hit["evidence_line"] - 1]
    assert "adapter" in line.lower()
    assert line.strip().startswith(hit["evidence_text"][:40])


# ------------------------------------------------------------------ evidence guards (a count cannot hide a defect)


def test_nothing_is_unclassified_and_no_persisted_write_lacks_a_version() -> None:
    report = build_closed_world_report()
    guards = _fx()["evidence_guards"]
    assert not report.unclassified_candidate_ids
    assert guards["unclassified_candidate_ids"] == 0
    kinds = [c.discovery_kind for c in report.raw_candidates]
    assert kinds.count("defect.persisted_without_version") == guards["defect_persisted_without_version_candidates"] == 6
    assert kinds.count("compat.shim") == guards["compat_shim_candidates"]
    assert guards["compat_shim_candidates"] == _fx()["previous"]["compat_shim_candidates"] + 4
    assert not any(row["kind"] == "defect.persisted_without_version" for row in _fx()["added_candidate_ids"])


def test_the_remainder_fingerprint_is_recomputed_from_the_current_inventory() -> None:
    assert _remainder_stable_inventory_fingerprint(build_closed_world_report()) == (
        REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256
    )


def test_the_current_head_inventory_gate_is_green_at_the_repinned_counters() -> None:
    assert validate_current_head_compat_closed_world_inventory() == ()


# ------------------------------------------------------------------ negative tests: the gates still bite


def test_hidden_removal_of_an_added_candidate_fails_the_inventory_gate() -> None:
    identity = _fx()["added_candidate_ids"][0]["id"]
    violations = validate_current_head_compat_closed_world_inventory(_report_without(identity))
    assert any("raw_candidates" in v for v in violations)


def test_one_extra_candidate_fails_the_inventory_gate() -> None:
    base = build_closed_world_report()
    extra = _synthetic_r3_inventory_probe_candidate("p7_extra")
    report = _closed_world_report_with_substituted_raw_candidates((*base.raw_candidates, extra))
    violations = validate_current_head_compat_closed_world_inventory(report)
    assert any("raw_candidates" in v for v in violations)


def test_a_balanced_substitution_of_an_added_candidate_fails_with_a_fingerprint_violation() -> None:
    base = build_closed_world_report()
    identity = _fx()["added_candidate_ids"][0]["id"]
    raw = tuple(c for c in base.raw_candidates if stable_inventory_candidate_identity(c) != identity)
    substituted = _closed_world_report_with_substituted_raw_candidates(
        (*raw, _synthetic_r3_inventory_probe_candidate("p7_substitute"))
    )
    assert len(substituted.raw_candidates) == ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES
    violations = validate_current_head_compat_closed_world_inventory(substituted)
    assert any("fingerprint" in v for v in violations)


def test_a_removed_candidate_returning_fails_the_inventory_gate() -> None:
    base = build_closed_world_report()
    row = _fx()["removed_candidate_ids"][0]
    revived = replace(
        _synthetic_r3_inventory_probe_candidate("p7_revived"),
        candidate_id=row["id"],
        semantic_identity=row["id"],
    )
    report = _closed_world_report_with_substituted_raw_candidates((*base.raw_candidates, revived))
    assert validate_current_head_compat_closed_world_inventory(report)


def test_the_shim_delta_gate_accepts_exactly_the_recorded_delta() -> None:
    accepted = load_replay_x_accepted_shim_delta()
    assert (
        shim_delta_accepted_r5_gate_violations(
            added_paths=accepted.added_paths, removed_paths=accepted.removed_paths, accepted=accepted
        )
        == ()
    )


def test_a_fifth_unexpected_shim_path_fails_the_shim_delta_gate() -> None:
    accepted = load_replay_x_accepted_shim_delta()
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=(*accepted.added_paths, "intergrax/p7_unexpected_shim.py"),
        removed_paths=accepted.removed_paths,
        accepted=accepted,
    )
    assert any("unexpected added shim paths" in v for v in violations)


def test_a_missing_recorded_shim_path_fails_the_shim_delta_gate() -> None:
    accepted = load_replay_x_accepted_shim_delta()
    hit = _fx()["shim_heuristic_hits"][0]["path"]
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=tuple(p for p in accepted.added_paths if p != hit),
        removed_paths=accepted.removed_paths,
        accepted=accepted,
    )
    assert any("mismatch" in v for v in violations)


def test_a_balanced_shim_path_substitution_fails_the_shim_delta_gate() -> None:
    accepted = load_replay_x_accepted_shim_delta()
    hit = _fx()["shim_heuristic_hits"][0]["path"]
    swapped = tuple(p for p in accepted.added_paths if p != hit) + ("intergrax/p7_substituted_shim.py",)
    assert len(swapped) == len(accepted.added_paths)
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=swapped, removed_paths=accepted.removed_paths, accepted=accepted
    )
    assert any("unexpected added shim paths" in v for v in violations)


def test_the_pins_are_module_attributes_of_the_reconciliation_owner() -> None:
    for name in (
        "REPLAY_X_ACCEPTED_REMAINDER_RAW_CANDIDATES",
        "REPLAY_X_ACCEPTED_REMAINDER_FINGERPRINT_SHA256",
        "ACCEPTED_CURRENT_HEAD_RAW_CANDIDATES",
        "ACCEPTED_CURRENT_HEAD_SEMANTIC_SURFACES",
        "ACCEPTED_CURRENT_HEAD_LEGACY_PROJECTED_SEMANTIC_SURFACES",
        "ACCEPTED_CURRENT_HEAD_EXCLUSIONS",
        "REPLAY_X_COMPAT_LINEAGE_HEAD_SHA",
        "PDC_X_P7_REPIN_MEASURED_AT_SHA",
    ):
        assert hasattr(recon, name), name
    assert recon.REPLAY_X_COMPAT_LINEAGE_HEAD_SHA == "05549f9acb9ecedf910c7470d442bba5d0a27337"
