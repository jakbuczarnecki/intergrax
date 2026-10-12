# © Artur Czarnecki. All rights reserved.

"""Pure in-memory R5 accepted shim delta reconciliation (no closed-world discovery)."""

from __future__ import annotations

from tests.qualification.compat_x._compat_x_deprecation_r5_shim_reconciliation import (
    AcceptedShimDeltaR5,
    load_accepted_shim_delta_r5,
    shim_delta_accepted_r5_gate_violations,
)

_ACCEPTED = load_accepted_shim_delta_r5()


def test_r5_shim_delta_accepted_exact_match_passes() -> None:
    assert (
        shim_delta_accepted_r5_gate_violations(
            added_paths=_ACCEPTED.added_paths,
            removed_paths=_ACCEPTED.removed_paths,
            accepted=_ACCEPTED,
        )
        == ()
    )


def test_r5_shim_delta_unexpected_third_addition_fails() -> None:
    added = _ACCEPTED.added_paths + ("intergrax/synthetic/third_unexpected_shim.py",)
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=added,
        removed_paths=_ACCEPTED.removed_paths,
        accepted=_ACCEPTED,
    )
    assert violations
    assert any("unexpected added" in v for v in violations)


def test_r5_shim_delta_historical_removal_fails() -> None:
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=_ACCEPTED.added_paths,
        removed_paths=("intergrax/agent_distribution/__init__.py",),
        accepted=_ACCEPTED,
    )
    assert violations
    assert any("removed historical" in v for v in violations)


def test_r5_shim_delta_balanced_substitution_fails() -> None:
    removed = ("intergrax/agent_distribution/__init__.py",)
    added = _ACCEPTED.added_paths + ("intergrax/synthetic/unexpected_substitution_shim.py",)
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=added,
        removed_paths=removed,
        accepted=_ACCEPTED,
    )
    assert len(violations) >= 2


def test_r5_shim_delta_count_only_match_still_fails_on_wrong_paths() -> None:
    wrong_added = (
        "intergrax/agent_distribution/sqlite_payload_legacy.py",
        "intergrax/synthetic/wrong_second_shim.py",
    )
    freeze = AcceptedShimDeltaR5(
        baseline_source_sha=_ACCEPTED.baseline_source_sha,
        added_paths=_ACCEPTED.added_paths,
        removed_paths=_ACCEPTED.removed_paths,
    )
    violations = shim_delta_accepted_r5_gate_violations(
        added_paths=wrong_added,
        removed_paths=(),
        accepted=freeze,
    )
    assert violations
    assert any("unexpected added" in v or "mismatch" in v for v in violations)
