# © Artur Czarnecki. All rights reserved.

"""EBH-7 task register SSOT (92 tasks, mirrors Correction R2 Annex B)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tests.qualification.ebh_7._constants import EBH7_TASKS_TOTAL


@dataclass(frozen=True, slots=True)
class Ebh7TaskRow:
    task_id: str
    wave: str
    owner: str
    action: str


def _build_tasks() -> tuple[Ebh7TaskRow, ...]:
    rows: list[Ebh7TaskRow] = []
    wave_specs: tuple[tuple[str, int, str, str], ...] = (
        ("W0", 10, "Program", "Entry/materialize SSOT"),
        ("W1", 5, "Architecture inventory", "Mechanism automation"),
        ("W2", 7, "Assurance planning", "Gap/history/risk"),
        ("W3", 10, "Execution/NPSC", "NPSC remediation"),
        ("W4", 8, "Governance", "Gov/DUP/SYS-INV L2"),
        ("W5", 10, "Integrations/VBN", "VBN artifacts + RPL-H"),
        ("W6", 6, "Tenant platform", "Tenant E2E"),
        ("W7", 7, "Cross-domain qual", "XDM scenarios"),
        ("W8", 8, "Platform qual", "ECAP C1-C5 + L3"),
        ("W9", 6, "Evidence", "Dossier/manifests"),
        ("W10", 5, "Independent audit", "Protocol 03"),
        ("W11", 10, "Operator program", "OD closure + EVOL handoff"),
    )
    n = 1
    for wave, count, owner, action in wave_specs:
        for _ in range(count):
            rows.append(Ebh7TaskRow(f"TASK-EBH7-{n:03d}", wave, owner, action))
            n += 1
    return tuple(rows)


EBH7_TASK_REGISTER: Final[tuple[Ebh7TaskRow, ...]] = _build_tasks()


def assert_ebh7_task_register_integrity() -> None:
    assert len(EBH7_TASK_REGISTER) == EBH7_TASKS_TOTAL
    ids = [r.task_id for r in EBH7_TASK_REGISTER]
    assert len(ids) == len(set(ids))
