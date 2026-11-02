# © Artur Czarnecki. All rights reserved.

"""EXP-S1: local agent / application blast radius (oracle O-F; metric M-06 escape count)."""

from __future__ import annotations

from testing_support.change_safety.assess.isolation import application_of_path, assess_isolation
from testing_support.change_safety.graph.impact import assess_path_change_impact
from tests.qualification.evol_x.experiments._harness import ArmResult, fixture_inventory, load_oracle

SCENARIO = "S1"


def _measure(inventory, changed: list[str]):
    paths = frozenset(changed)
    impact = assess_path_change_impact(inventory, changed_paths=paths)
    return assess_isolation(impact, changed_paths=paths)


def naive_blast_apps(changed: list[str]) -> tuple[str, ...]:
    """Filename-prefix baseline: only the application owning the changed path (misses cross-app consumers)."""
    return tuple(sorted({a for a in (application_of_path(p) for p in changed) if a}))


def run() -> tuple[ArmResult, ArmResult]:
    oracle = load_oracle("s1_local_agent")["scenarios"]
    with fixture_inventory("s1_local_agent") as inventory:
        local = _measure(inventory, oracle["local"]["changed"])
        escape = _measure(inventory, oracle["escape"]["changed"])
    pos_ok = (
        list(local.blast_apps) == oracle["local"]["expected_blast_apps"]
        and list(local.escapes) == oracle["local"]["expected_escapes"]
        and local.isolated
    )
    naive = list(naive_blast_apps(oracle["escape"]["changed"]))
    neg_ok = (
        list(escape.blast_apps) == oracle["escape"]["expected_blast_apps"]
        and list(escape.escapes) == oracle["escape"]["expected_escapes"]
        and not escape.isolated
        and naive != oracle["escape"]["expected_blast_apps"]  # non-vacuity: baseline would have missed the escape
    )
    return (
        ArmResult(SCENARIO, "positive", pos_ok, f"blast_apps={list(local.blast_apps)} escapes={list(local.escapes)}"),
        ArmResult(
            SCENARIO,
            "negative",
            neg_ok,
            f"escapes={list(escape.escapes)} naive_blast={naive} blast={list(escape.blast_apps)}",
        ),
    )
