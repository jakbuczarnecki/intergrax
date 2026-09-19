# © Artur Czarnecki. All rights reserved.

"""Parse EAC-1 §4.1 peer authority register from canonical markdown (SSOT)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_EAC1 = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "audits"
    / "ENTERPRISE_CROSS_LAYER_RESPONSIBILITY_OWNERSHIP_MATRIX_EAC1.md"
)

_PEER_SECTION_HEADING = "## 4.1 Strict peer authority register"
_DOMAIN_ROLE_SECTION_MARKER = "### 4.B DOMAIN ROLE"
_SUBORDINATE_SECTION_MARKER = "### 4.C SUBORDINATE / INTERNAL AUTHORITY"
_PEER_COVERAGE_SECTION_MARKER = "### 5.2 Peer authority coverage"
_DECLARED_COUNT_RE = re.compile(
    r"CURRENT_PEER_AUTHORITY_COUNT\s*=\s*(\d+)",
    re.MULTILINE,
)
_DEPRECATED_LABELS_RE = re.compile(
    r"Deprecated authority labels \(EAC-1R2[^:]*:\s*(.+?)\.",
    re.DOTALL,
)
_BACKTICK_LABEL_RE = re.compile(r"`([^`]+)`")

_AMBIGUOUS_OWNER_RE = re.compile(
    r"^(?P<a>[A-Z][A-Z0-9_]+(?:\s*\([^)]+\))?)"
    r"\s*(?:/|&|\band\b)\s*"
    r"(?P<b>[A-Z][A-Z0-9_]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PeerAuthorityRow:
    authority_type: str
    canonical_owner: str
    competing_peer_owner: str


@dataclass(frozen=True, slots=True)
class SubordinateAuthorityRow:
    authority_type: str
    internal_owner: str


def eac1_text() -> str:
    return _EAC1.read_text(encoding="utf-8-sig")


def _section_until_next_h2(text: str, start_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"Missing section marker: {start_marker}")
    rest = text[start + len(start_marker) :]
    match = re.search(r"\n## ", rest)
    if match:
        return rest[: match.start()]
    return rest


def _section_until_next_heading(text: str, start_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        raise ValueError(f"Missing section marker: {start_marker}")
    rest = text[start + len(start_marker) :]
    match = re.search(r"\n### |\n## ", rest)
    if match:
        return rest[: match.start()]
    return rest


def _strip_md(cell: str) -> str:
    value = cell.strip()
    value = re.sub(r"\*\*([^*]+)\*\*", r"\1", value)
    value = value.replace("\\", "")
    return value.strip()


def _parse_table(section: str) -> list[list[str]]:
    lines = [line for line in section.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return []
    rows: list[list[str]] = []
    for line in lines:
        if re.search(r"\|\s*[-:]+", line):
            continue
        cells = [_strip_md(part) for part in line.split("|")[1:-1]]
        if not any(cells):
            continue
        if cells[0].lower() in {
            "authority type",
            "subordinate authority type",
        }:
            continue
        rows.append(cells)
    return rows


def parse_peer_authority_register(text: str | None = None) -> list[PeerAuthorityRow]:
    body = text if text is not None else eac1_text()
    section = _section_until_next_h2(body, _PEER_SECTION_HEADING)
    parsed: list[PeerAuthorityRow] = []
    for cells in _parse_table(section):
        if len(cells) < 5:
            continue
        parsed.append(
            PeerAuthorityRow(
                authority_type=cells[0],
                canonical_owner=cells[1],
                competing_peer_owner=cells[4] if len(cells) > 4 else "",
            )
        )
    return parsed


def parse_subordinate_authority_register(
    text: str | None = None,
) -> list[SubordinateAuthorityRow]:
    body = text if text is not None else eac1_text()
    section = _section_until_next_heading(body, _SUBORDINATE_SECTION_MARKER)
    parsed: list[SubordinateAuthorityRow] = []
    for cells in _parse_table(section):
        if len(cells) < 2:
            continue
        parsed.append(
            SubordinateAuthorityRow(
                authority_type=cells[0],
                internal_owner=cells[1],
            )
        )
    return parsed


def parse_declared_peer_authority_count(text: str | None = None) -> int:
    body = text if text is not None else eac1_text()
    match = _DECLARED_COUNT_RE.search(body)
    if not match:
        raise ValueError("CURRENT_PEER_AUTHORITY_COUNT not found in EAC-1")
    return int(match.group(1))


def parse_deprecated_authority_labels(text: str | None = None) -> frozenset[str]:
    body = text if text is not None else eac1_text()
    match = _DEPRECATED_LABELS_RE.search(body)
    if not match:
        raise ValueError("Deprecated authority labels line not found in EAC-1")
    labels = _BACKTICK_LABEL_RE.findall(match.group(1))
    return frozenset(labels)


def canonical_owner_is_singular(owner: str) -> bool:
    cleaned = owner.strip()
    if not cleaned or cleaned in {"—", "-", "NONE"}:
        return False
    if _AMBIGUOUS_OWNER_RE.match(cleaned):
        return False
    return True


def peer_authority_type_names(rows: list[PeerAuthorityRow]) -> list[str]:
    return [row.authority_type for row in rows]


def parse_domain_role_register(text: str | None = None) -> list[str]:
    body = text if text is not None else eac1_text()
    section = _section_until_next_heading(body, _DOMAIN_ROLE_SECTION_MARKER)
    names: list[str] = []
    for cells in _parse_table(section):
        if not cells:
            continue
        names.append(cells[0])
    return names


def count_peer_authorities(rows: list[PeerAuthorityRow]) -> int:
    return len(rows)


def count_peer_types_with_single_canonical_owner(rows: list[PeerAuthorityRow]) -> int:
    return sum(
        1
        for row in rows
        if row.canonical_owner.strip()
        and row.canonical_owner.strip() not in {"—", "-", "NONE"}
        and canonical_owner_is_singular(row.canonical_owner)
    )


def _competing_peer_owner_is_second_canonical(cell: str) -> bool:
    cleaned = cell.strip()
    if not cleaned or cleaned in {"—", "-"}:
        return False
    upper = cleaned.upper()
    if upper == "NONE" or upper.startswith("NONE "):
        return False
    if "ADR REQUIRED" in upper:
        return False
    if upper.startswith("PASS"):
        return False
    return True


def row_has_competing_canonical_owner(row: PeerAuthorityRow) -> bool:
    if not canonical_owner_is_singular(row.canonical_owner):
        return True
    return _competing_peer_owner_is_second_canonical(row.competing_peer_owner)


def count_peer_types_with_competing_canonical_owner(rows: list[PeerAuthorityRow]) -> int:
    return sum(1 for row in rows if row_has_competing_canonical_owner(row))


def count_subordinate_authorities(rows: list[SubordinateAuthorityRow]) -> int:
    return len(rows)


def count_domain_role_types(text: str | None = None) -> int:
    return len(parse_domain_role_register(text))


def parse_peer_authority_coverage_metrics(text: str | None = None) -> dict[str, int]:
    body = text if text is not None else eac1_text()
    section = _section_until_next_heading(body, _PEER_COVERAGE_SECTION_MARKER)
    metrics: dict[str, int] = {}
    for cells in _parse_table(section):
        if len(cells) < 2:
            continue
        value_cell = cells[1].strip()
        if not value_cell.isdigit():
            continue
        metrics[cells[0].strip()] = int(value_cell)
    return metrics


_COVERAGE_PEER_TYPES_KEY = "Peer authority types (§4.A)"
_COVERAGE_DOMAIN_ROLES_KEY = "Domain role types (§4.B)"
_COVERAGE_SUBORDINATE_KEY = "Subordinate authority types (§4.C)"
_COVERAGE_SINGLE_OWNER_KEY = "Peer types with exactly one canonical owner"
_COVERAGE_COMPETING_KEY = "Peer types with competing canonical owners"
