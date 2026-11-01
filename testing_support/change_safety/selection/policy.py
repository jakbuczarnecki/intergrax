# © Artur Czarnecki. All rights reserved.

"""Policy II-A ``EVOL_CONSERVATIVE_FULL`` (EVOL-X scoped); II-B ``PLATFORM_CONSERVATIVE_FULL`` is never claimed."""

from __future__ import annotations

from dataclasses import dataclass

from testing_support.change_safety.data.loaders import load_evol_x_data_document

POLICY_ID = "EVOL_CONSERVATIVE_FULL"
REASON_RULE_SCOPED = "EVOL-R7-SCOPED"
FORBIDDEN_CLAIM_POLICY = "PLATFORM_CONSERVATIVE_FULL"


def selection_policy_forbid_empty() -> bool:
    policy = load_evol_x_data_document("selection_policy.v1.json")
    return bool(policy.get("forbid_empty_selection"))


@dataclass(frozen=True)
class ConservativeExclusions:
    node_prefixes: frozenset[str]
    populations: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    def excludes(self, node_id: str) -> bool:
        return any(node_id == p or node_id.startswith(p + "::") for p in self.node_prefixes)


def load_conservative_exclusions() -> ConservativeExclusions:
    doc = load_evol_x_data_document("evol_conservative_exclusions.v1.json")
    rows = tuple(doc["exclusions"])
    return ConservativeExclusions(
        node_prefixes=frozenset(r["module_or_node"] for r in rows),
        populations=tuple(doc.get("excluded_populations", ())),
        rows=rows,
    )


def evol_conservative_full_set(
    *,
    t1: frozenset[str],
    t2: frozenset[str],
    t3: frozenset[str],
    adjacent: frozenset[str] = frozenset(),
    exclusions: ConservativeExclusions | None = None,
) -> frozenset[str]:
    """``UNION(T1,T2,T3,Adj) \\ exclusions`` (Part II-A)."""
    union = t1 | t2 | t3 | adjacent
    if exclusions is None:
        return union
    return frozenset(n for n in union if not exclusions.excludes(n))
