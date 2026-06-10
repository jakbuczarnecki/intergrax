# © Artur Czarnecki. All rights reserved.

"""Architecture debt burn-down tied to AUDIT-IDEAL milestones (AUDIT-IDEAL-32.1)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_DONE_AUDIT_PATTERN = re.compile(r"AUDIT-IDEAL-([\w.]+).*\|\s*\*\*Done\*\*")
_DEBT_ROW_PATTERN = re.compile(
    r"\|\s*(DEBT-[\w-]+)\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(AUDIT-IDEAL-[\w.]+)\s*\|\s*([^|]+)\s*\|"
)


@dataclass(frozen=True, slots=True)
class DebtBurnDownRecord:
    debt_id: str
    audit_ideal_id: str
    status: str


@dataclass(frozen=True, slots=True)
class DebtBurnDownReport:
    done_audit_ids: tuple[str, ...]
    records: tuple[DebtBurnDownRecord, ...]
    unresolved_debt_ids: tuple[str, ...]


def parse_debt_register(register_text: str) -> tuple[DebtBurnDownRecord, ...]:
    records: list[DebtBurnDownRecord] = []
    for match in _DEBT_ROW_PATTERN.finditer(register_text):
        records.append(
            DebtBurnDownRecord(
                debt_id=match.group(1),
                audit_ideal_id=match.group(2),
                status=match.group(3).strip(),
            )
        )
    return tuple(records)


def parse_done_audit_ideal_ids(audit_register_text: str) -> tuple[str, ...]:
    return tuple(sorted({match.group(1) for match in _DONE_AUDIT_PATTERN.finditer(audit_register_text)}))


def build_debt_burn_down_report(
    *,
    debt_register_text: str,
    audit_register_text: str,
) -> DebtBurnDownReport:
    """Verify debt rows linked to Done AUDIT-IDEAL items are marked Closed."""
    done_ids = parse_done_audit_ideal_ids(audit_register_text)
    records = parse_debt_register(debt_register_text)
    done_set = set(done_ids)
    unresolved: list[str] = []
    for record in records:
        audit_suffix = record.audit_ideal_id.removeprefix("AUDIT-IDEAL-")
        if audit_suffix not in done_set:
            continue
        if "Closed" not in record.status:
            unresolved.append(record.debt_id)
    return DebtBurnDownReport(
        done_audit_ids=done_ids,
        records=records,
        unresolved_debt_ids=tuple(unresolved),
    )


def load_debt_burn_down_report(repo_root: Path) -> DebtBurnDownReport:
    debt_path = repo_root / "docs" / "guides" / "ARCHITECTURE_DEBT_REGISTER.md"
    audit_path = repo_root / "docs" / "plan" / "AUDIT_IDEAL_2026.md"
    return build_debt_burn_down_report(
        debt_register_text=debt_path.read_text(encoding="utf-8"),
        audit_register_text=audit_path.read_text(encoding="utf-8"),
    )
