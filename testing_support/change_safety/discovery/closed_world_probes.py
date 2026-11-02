# © Artur Czarnecki. All rights reserved.

"""CW-07: ≥10 mutation probe classes; each must flip its target gate from PASS to FAIL."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

from testing_support.change_safety.assess.consumers import critical_text_pairs
from testing_support.change_safety.discovery.closed_world import ClosedWorldInventory
from testing_support.change_safety.discovery.closed_world_gates import (
    ClosedWorldGateCheck,
    run_cw01,
    run_cw02,
    run_cw03,
    run_cw04,
    run_cw05,
    run_cw06,
    run_cw08,
)
from testing_support.change_safety.discovery.extractors import GraphEdge


@dataclass(frozen=True)
class ProbeInputs:
    inventory: ClosedWorldInventory
    owner_rules: dict[str, Any]
    exclusion_register: dict[str, Any]
    collected_node_ids: frozenset[str]


@dataclass(frozen=True)
class ProbeResult:
    probe_id: str
    gate_id: str
    baseline_passed: bool
    mutated_passed: bool

    @property
    def flipped(self) -> bool:
        return self.baseline_passed and not self.mutated_passed


@dataclass(frozen=True)
class Probe:
    probe_id: str
    gate_id: str
    run: Callable[[ProbeInputs], ClosedWorldGateCheck]


def _first_module_path(inv: ClosedWorldInventory) -> str:
    return next(n.path for n in inv.nodes if n.category == "module")


def _p01(i: ProbeInputs) -> ClosedWorldGateCheck:
    inv = replace(i.inventory, universe_paths=i.inventory.universe_paths + ("intergrax/__probe_unclassified__.py",))
    return run_cw01(inv, i.exclusion_register)


def _p02(i: ProbeInputs) -> ClosedWorldGateCheck:
    victim = _first_module_path(i.inventory)
    inv = replace(i.inventory, nodes=tuple(n for n in i.inventory.nodes if n.path != victim))
    return run_cw01(inv, i.exclusion_register)


def _p03(i: ProbeInputs) -> ClosedWorldGateCheck:
    rules = copy.deepcopy(i.owner_rules)
    rules["owners"].append({"owner_id": "PROBE-DUP", "domain": "probe", "paths_prefix": ["intergrax/"]})
    return run_cw02(i.inventory, rules)


def _p04(i: ProbeInputs) -> ClosedWorldGateCheck:
    rules = copy.deepcopy(i.owner_rules)
    rules["owners"] = [o for o in rules["owners"] if o["owner_id"] != "EVOL-X-A"]
    return run_cw02(i.inventory, rules)


def _p05(i: ProbeInputs) -> ClosedWorldGateCheck:
    src = _first_module_path(i.inventory)
    edge = GraphEdge("module:" + src[:-3].replace("/", "."), "module:intergrax.__probe_missing__", "IMPORTS")
    return run_cw03(i.inventory.with_edges(i.inventory.edges + (edge,)))


def _p06(i: ProbeInputs) -> ClosedWorldGateCheck:
    inv = replace(
        i.inventory,
        unresolved_imports=i.inventory.unresolved_imports + (("module:intergrax.x", "intergrax.__silent__"),),
    )
    return run_cw03(inv)


def _p07(i: ProbeInputs) -> ClosedWorldGateCheck:
    reg = copy.deepcopy(i.exclusion_register)
    reg["entries"].append(
        {"exclusion_id": "X-PROBE", "status": "OPEN", "predicate": "HISTORICAL_DOC", "glob_patterns": []}
    )
    return run_cw04(i.inventory, reg)


def _p08(i: ProbeInputs) -> ClosedWorldGateCheck:
    inv = replace(i.inventory, k_to_x_edges=i.inventory.k_to_x_edges + (("module:intergrax.x", "docs.thing"),))
    return run_cw04(inv, i.exclusion_register)


def _p09(i: ProbeInputs) -> ClosedWorldGateCheck:
    victim = next(n for n in sorted(i.collected_node_ids) if "[" not in n)
    return run_cw05(i.inventory, collected_node_ids=i.collected_node_ids - {victim})


def _p10(i: ProbeInputs) -> ClosedWorldGateCheck:
    extra = "tests/qualification/evol_x/gates/test_probe_phantom.py::test_phantom"
    return run_cw05(i.inventory, collected_node_ids=i.collected_node_ids | {extra})


def _p11(i: ProbeInputs) -> ClosedWorldGateCheck:
    """Leave-one-out: drop one contract consumer's edges; second method must report an omission."""
    victim = critical_text_pairs(i.inventory)[0]
    edges = tuple(e for e in i.inventory.edges if not (e.source_id == victim[0] and e.target_id == victim[1]))
    return run_cw06(i.inventory.with_edges(edges))


def _p12(i: ProbeInputs) -> ClosedWorldGateCheck:
    mutated = i.inventory.with_edges(i.inventory.edges[:-1])
    return run_cw08(i.inventory, mutated)


PROBES: tuple[Probe, ...] = (
    Probe("P01-unclassified-candidate", "CW-01", _p01),
    Probe("P02-dropped-module-node", "CW-01", _p02),
    Probe("P03-duplicate-owner", "CW-02", _p03),
    Probe("P04-missing-owner", "CW-02", _p04),
    Probe("P05-dangling-import-edge", "CW-03", _p05),
    Probe("P06-silent-dangling-import", "CW-03", _p06),
    Probe("P07-open-exclusion", "CW-04", _p07),
    Probe("P08-k-to-x-edge", "CW-04", _p08),
    Probe("P09-collect-missing-vs-ast", "CW-05", _p09),
    Probe("P10-collect-phantom-vs-ast", "CW-05", _p10),
    Probe("P11-leave-one-out-consumer", "CW-06", _p11),
    Probe("P12-digest-drift", "CW-08", _p12),
)

_BASELINE: dict[str, Callable[[ProbeInputs], ClosedWorldGateCheck]] = {
    "CW-01": lambda i: run_cw01(i.inventory, i.exclusion_register),
    "CW-02": lambda i: run_cw02(i.inventory, i.owner_rules),
    "CW-03": lambda i: run_cw03(i.inventory),
    "CW-04": lambda i: run_cw04(i.inventory, i.exclusion_register),
    "CW-05": lambda i: run_cw05(i.inventory, collected_node_ids=i.collected_node_ids),
    "CW-06": lambda i: run_cw06(i.inventory),
    "CW-08": lambda i: run_cw08(i.inventory, i.inventory),
}


def run_mutation_probes(inputs: ProbeInputs) -> tuple[ProbeResult, ...]:
    baselines = {gate: fn(inputs).passed for gate, fn in _BASELINE.items()}
    return tuple(ProbeResult(p.probe_id, p.gate_id, baselines[p.gate_id], p.run(inputs).passed) for p in PROBES)


def run_cw07(inputs: ProbeInputs) -> ClosedWorldGateCheck:
    results = run_mutation_probes(inputs)
    flipped = sum(1 for r in results if r.flipped)
    classes = len({r.gate_id for r in results if r.flipped})
    passed = flipped == len(results) and len(results) >= 10
    return ClosedWorldGateCheck(
        "CW-07",
        passed,
        f"{flipped}/{len(results)} mutation probes flip a gate" if passed else "mutation probe missed",
        {"probes": len(results), "flipped": flipped, "gate_classes_flipped": classes},
    )
