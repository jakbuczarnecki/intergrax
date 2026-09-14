# © Artur Czarnecki. All rights reserved.

"""Shared canonical Execution / Decision documentation gate helpers (EE-POST-FREEZE-FINAL-R1)."""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]

MAINTAINER_HUB = (
    _REPO_ROOT / "docs/project/maintainers/architecture/EXECUTION_ENGINE.md"
)
FINAL_ENTERPRISE_ARCHITECTURE = (
    _REPO_ROOT
    / "docs/project/maintainers/architecture/EXECUTION_ENGINE_FINAL_ENTERPRISE_ARCHITECTURE.md"
)
DOCUMENTATION_INVENTORY = (
    _REPO_ROOT
    / "docs/project/maintainers/architecture/EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md"
)
DECISION_SYSTEM = _REPO_ROOT / "docs/project/architecture/DECISION_SYSTEM.md"
DECISION_SYSTEM_ARCHITECTURE = (
    _REPO_ROOT / "docs/project/architecture/DECISION_SYSTEM_ARCHITECTURE.md"
)
DECISION_SYSTEM_EXTENDED = (
    _REPO_ROOT
    / "docs/project/architecture/satellites/DECISION_SYSTEM_extended_depth.md"
)
DECISION_SYSTEM_PLAN = _REPO_ROOT / "docs/project/maintainers/plans/DECISION_SYSTEM.md"
POST_FREEZE_GAP_AUDIT = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/EXECUTION_ENGINE_POST_FREEZE_EXHAUSTIVE_GAP_AUDIT.md"
)
DOC_RECONCILIATION = (
    _REPO_ROOT
    / "docs/project/maintainers/qualification/EXECUTION_ENGINE_AND_DECISION_DOCUMENTATION_RECONCILIATION.md"
)

CANONICAL_EXECUTION_DECISION_DOCS: tuple[Path, ...] = (
    MAINTAINER_HUB,
    FINAL_ENTERPRISE_ARCHITECTURE,
    DOCUMENTATION_INVENTORY,
    DECISION_SYSTEM,
    DECISION_SYSTEM_ARCHITECTURE,
    DECISION_SYSTEM_EXTENDED,
    DECISION_SYSTEM_PLAN,
    POST_FREEZE_GAP_AUDIT,
    DOC_RECONCILIATION,
)

# cp1252 mis-decode of UTF-8 punctuation (longest tokens first)
_MOJIBAKE_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("\u00e2\u201d\u0153", "\u251c"),
    ("\u00e2\u201d\u015b", "\u251c"),
    ("\u00e2\u201d\u20ac", "\u2500"),
    ("\u00e2\u201d\u201d", "\u2514"),
    ("\u00e2\u201d\u201a", "\u2502"),
    ("\u00e2\u20ac\u015b", "\u201c"),
    ("\u00e2\u20ac\u0165", "\u201d"),
    ("\u00e2\u20ac\u2122", "\u2019"),
    ("\u00e2\u20ac\u00a6", "\u2026"),
    ("\u00e2\u20ac\u201d", "\u2014"),
    ("\u00e2\u20ac\u201c", "\u2013"),
    ("\u00e2\u2020\u2019", "\u2192"),
    ("\u00e2\u2020\u201c", "\u2193"),
    ("\u00e2\u2020\u201d", "\u2194"),
    ("\u00e2\u2030\u0104", "\u2265"),
    ("\u00e2\u2030\u00a5", "\u2265"),
    ("\u00e2\u2030\u00a4", "\u2264"),
    ("\u00e2\u2030\u00a0", "\u00a0"),
    ("\u00c2\u00b7", "\u00b7"),
    ("\u00c2\u00a7", "\u00a7"),
)

MOJIBAKE_PATTERN = re.compile(
    r"\u00e2\u20ac|\u00c2\u00a7|\u00c2\u00b7|\u00e2\u2020|\u00e2\u201d|\u00e2\u2030|\u00c3"
)

FORBIDDEN_STALE_CURRENT_STATE_CLAIMS: tuple[str, ...] = (
    "await R2 Final qualification/freeze",
    "R2 implemented, not final-frozen",
    "await R2 Final freeze",
    "implementation complete — await R2 Final",
    "R2 pending final freeze",
)

NPSC5F_CURRENT_STATUS_MARKERS: tuple[str, ...] = (
    "NPSC-5F/R1–R4",
    "NPSC-5F Final",
    "FROZEN / PASS",
    "RE-FROZEN",
    "REQUALIFIED",
    "EE-FINAL-02",
)


def read_doc(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    return raw.decode("utf-8")


def repair_mojibake(text: str) -> str:
    out = text
    for bad, good in _MOJIBAKE_REPLACEMENTS:
        out = out.replace(bad, good)
    return out


def count_mojibake(text: str) -> int:
    return len(MOJIBAKE_PATTERN.findall(text))


def find_mojibake_violations() -> list[str]:
    violations: list[str] = []
    for path in CANONICAL_EXECUTION_DECISION_DOCS:
        if not path.is_file():
            violations.append(f"missing: {path.relative_to(_REPO_ROOT)}")
            continue
        text = read_doc(path)
        count = count_mojibake(text)
        if count:
            violations.append(
                f"{path.relative_to(_REPO_ROOT)}: {count} mojibake token(s)"
            )
    return violations


def find_stale_current_state_claims() -> list[str]:
    violations: list[str] = []
    hub = read_doc(MAINTAINER_HUB)
    for claim in FORBIDDEN_STALE_CURRENT_STATE_CLAIMS:
        if claim in hub:
            violations.append(f"EXECUTION_ENGINE.md: {claim!r}")
    inv = read_doc(DOCUMENTATION_INVENTORY)
    for claim in FORBIDDEN_STALE_CURRENT_STATE_CLAIMS:
        if claim in inv:
            violations.append(f"EXECUTION_ENGINE_DOCUMENTATION_INVENTORY.md: {claim!r}")
    return violations
