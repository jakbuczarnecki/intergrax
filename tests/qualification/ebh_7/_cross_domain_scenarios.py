# © Artur Czarnecki. All rights reserved.

"""Annex F — 34 cross-domain scenarios (registry SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_7._constants import EBH7_CROSS_DOMAIN_COUNT


@dataclass(frozen=True, slots=True)
class CrossDomainScenario:
    scenario_id: str
    reuse_program: str
    evidence_id: str


EBH7_CROSS_DOMAIN_SCENARIOS: Final[tuple[CrossDomainScenario, ...]] = tuple(
    CrossDomainScenario(
        scenario_id=f"EBH7-XDM-{n:03d}",
        reuse_program="integrated qualification reuse @ HEAD",
        evidence_id=f"EVID-EBH7-XDM-{n:03d}",
    )
    for n in range(1, EBH7_CROSS_DOMAIN_COUNT + 1)
)


def assert_ebh7_cross_domain_registry_integrity() -> None:
    assert len(EBH7_CROSS_DOMAIN_SCENARIOS) == EBH7_CROSS_DOMAIN_COUNT
    ids = [s.scenario_id for s in EBH7_CROSS_DOMAIN_SCENARIOS]
    assert ids == [f"EBH7-XDM-{n:03d}" for n in range(1, EBH7_CROSS_DOMAIN_COUNT + 1)]
