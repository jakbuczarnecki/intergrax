# © Artur Czarnecki. All rights reserved.

"""S1..S8 experiment arm implementations (shared by test modules and the W6 verdict collector)."""

from __future__ import annotations

from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from tests.qualification.evol_x.experiments._harness import ArmResult
from tests.qualification.evol_x.experiments.arms import s1, s2, s3, s4, s5, s6, s7, s8

SCENARIO_IDS = tuple(f"S{i}" for i in range(1, 9))


def run_all(inventory: ClosedWorldInventory) -> tuple[ArmResult, ...]:
    """16 arm results (8 scenarios x positive/negative)."""
    results: list[ArmResult] = []
    for module in (s1, s2, s3, s4, s5, s7, s8):
        results.extend(module.run())
    results.extend(s6.run(inventory))
    return tuple(sorted(results, key=lambda r: (r.scenario_id, r.arm)))
