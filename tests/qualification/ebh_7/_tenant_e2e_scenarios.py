# © Artur Czarnecki. All rights reserved.

"""Annex E — 26 platform tenant E2E scenarios (registry SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_7._constants import EBH7_TENANT_E2E_COUNT


@dataclass(frozen=True, slots=True)
class TenantE2EScenario:
    scenario_id: str
    reuse_program: str
    evidence_id: str


EBH7_TENANT_E2E_SCENARIOS: Final[tuple[TenantE2EScenario, ...]] = tuple(
    TenantE2EScenario(
        scenario_id=f"EBH7-TEN-E2E-{n:02d}",
        reuse_program="tenant_x HEAD qualification suite",
        evidence_id=f"EVID-EBH7-TEN-E2E-{n:02d}",
    )
    for n in range(1, EBH7_TENANT_E2E_COUNT + 1)
)


def assert_ebh7_tenant_e2e_registry_integrity() -> None:
    assert len(EBH7_TENANT_E2E_SCENARIOS) == EBH7_TENANT_E2E_COUNT
    ids = [s.scenario_id for s in EBH7_TENANT_E2E_SCENARIOS]
    assert ids == [f"EBH7-TEN-E2E-{n:02d}" for n in range(1, EBH7_TENANT_E2E_COUNT + 1)]
