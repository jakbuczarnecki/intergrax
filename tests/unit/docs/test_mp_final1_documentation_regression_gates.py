# © Artur Czarnecki. All rights reserved.

"""MP-FINAL-1 — Multiplayer SSOT maturity + visual architecture documentation gates."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MP_ARCH = (
    _REPO_ROOT / "docs" / "project" / "capabilities" / "architecture" / "MULTIPLAYER_AI.md"
)
_MP_PLAN = _REPO_ROOT / "docs" / "project" / "capabilities" / "plan" / "MULTIPLAYER_AI.md"
_CW_ARCH = _REPO_ROOT / "docs" / "project" / "architecture" / "COLLABORATIVE_WORK.md"
_ROADMAP = _REPO_ROOT / "docs" / "project" / "overview" / "ROADMAP.md"
_ARCH_OVERVIEW = _REPO_ROOT / "docs" / "project" / "architecture" / "ARCHITECTURE_OVERVIEW.md"
_PUBLIC_MAP = _REPO_ROOT / "docs" / "project" / "community" / "PUBLIC_DOCUMENTATION_MAP.md"

_REQUIRED_STATUS_MARKERS_ARCH = (
    "Current Enterprise Maturity Boundary",
    "ENTERPRISE CORE IMPLEMENTED",
    "CAPABILITY EXPANSION PLANNED",
    "MP-5 — ENTERPRISE CERTIFIED / CLOSED",
    "MP-6 — ENTERPRISE CERTIFIED / CLOSED",
    "MP-7 — ENTERPRISE BOUNDARY CERTIFIED / CLOSED",
    "Multiplayer Tier-3 consumability boundary certified",
    "PLANNED / NOT STARTED",
    "Operability / Diagnostics Boundary",
    "Product / Visual UX Boundary",
    "External Agent Boundary",
)

_REQUIRED_STATUS_MARKERS_PLAN = (
    "MP-5 — ENTERPRISE CERTIFIED / CLOSED",
    "MP-6 — ENTERPRISE CERTIFIED / CLOSED",
    "ENTERPRISE BOUNDARY CERTIFIED / CLOSED",
    "PLANNED / NOT STARTED",
    "MP-FINAL-1",
    "MP-FINAL-2",
)

_REQUIRED_DIAGRAM_HEADINGS = (
    "### Diagram 1 — Capability ownership map",
    "### Diagram 2 — Layer / dependency architecture",
    "### Diagram 3 — Authority / mutation path",
    "### Diagram 4 — Shared work lifecycle relation",
    "### Diagram 5 — ContextView composition",
    "### Diagram 6 — Collaborative Activity",
    "### Diagram 7 — Tier-3 consumption / MP-7",
    "### Diagram 8 — Capability maturity",
)

_REQUIRED_LINKS = (
    "MP-5H_FINAL_ENTERPRISE_CERTIFICATION.md",
    "MP-6_FINAL_ENTERPRISE_CERTIFICATION.md",
    "MP-7D_FINAL_REFERENCE_CONSUMER_BOUNDARY_ENTERPRISE_CERTIFICATION.md",
    "ADR-MP-006",
    "ADR-MP-007",
    "ADR-MP-008",
)

_STALE_CURRENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "MP-5F…MP-9 remain roadmap",
        re.compile(r"MP-5F…MP-9 remain roadmap"),
    ),
    (
        "MP-7 NEXT as current",
        re.compile(r"MP-7[^\n]{0,60}NEXT"),
    ),
    (
        "MP-5F NEXT as current",
        re.compile(r"\*\*MP-5F — NEXT\*\*"),
    ),
    (
        "runtime proof not established (misleading)",
        re.compile(
            r"[Mm]ultiplayer[^\n]{0,160}runtime proof[^\n]{0,40}not (?:yet )?established"
        ),
    ),
)

_OWNERSHIP_REGRESSION = (
    re.compile(r"Multiplayer owns (?:canonical )?Decision", re.I),
    re.compile(r"ContextView owns (?:Memory|RAG|source truth)", re.I),
    re.compile(r"LKW owns Principal", re.I),
    re.compile(r"CollaborativeActivity\s*==\s*Observability", re.I),
    re.compile(r"Activity == observability", re.I),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _current_state_windows(text: str) -> list[str]:
    """Return non-historical occurrences of forbidden current-state stale claims."""
    violations: list[str] = []
    for label, pattern in _STALE_CURRENT_PATTERNS:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 120)
            window = text[start : match.end() + 40]
            if re.search(r"(?i)historical|superseded|at time of", window):
                continue
            violations.append(f"{label}: …{window!r}…")
    return violations


def test_canonical_maturity_ssot_markers() -> None:
    text = _read(_MP_ARCH)
    missing = [m for m in _REQUIRED_STATUS_MARKERS_ARCH if m not in text]
    assert not missing, f"architecture maturity SSOT missing: {missing}"


def test_plan_delivery_status_aligned() -> None:
    text = _read(_MP_PLAN)
    missing = [m for m in _REQUIRED_STATUS_MARKERS_PLAN if m not in text]
    assert not missing, f"plan status markers missing: {missing}"
    assert "MP-8" in text and "PLANNED / NOT STARTED" in text
    assert "MP-9" in text and "PLANNED / NOT STARTED" in text


def test_visual_architecture_diagrams_present() -> None:
    text = _read(_MP_ARCH)
    assert "## Visual architecture (MP-FINAL-1)" in text
    missing = [h for h in _REQUIRED_DIAGRAM_HEADINGS if h not in text]
    assert not missing, f"missing diagram headings: {missing}"
    assert text.count("```mermaid") >= 8


def test_no_stale_current_state_claims_in_canonical_docs() -> None:
    docs = {
        "multiplayer_architecture": _MP_ARCH,
        "multiplayer_plan": _MP_PLAN,
        "collaborative_work": _CW_ARCH,
        "roadmap": _ROADMAP,
        "architecture_overview": _ARCH_OVERVIEW,
        "public_documentation_map": _PUBLIC_MAP,
    }
    violations: list[str] = []
    for name, path in docs.items():
        for item in _current_state_windows(_read(path)):
            violations.append(f"{name}: {item}")
    assert not violations, "stale current-state claims:\n" + "\n".join(violations)


def test_no_ownership_regression_language() -> None:
    text = _read(_MP_ARCH)
    hits = [p.pattern for p in _OWNERSHIP_REGRESSION if p.search(text)]
    assert not hits, f"ownership regression language: {hits}"


def test_mp7_semantics_not_lkw_adoption_complete() -> None:
    text = _read(_MP_ARCH)
    assert "Multiplayer Tier-3 consumability boundary certified" in text
    assert "not** LKW Multiplayer product adoption complete" in text
    for match in re.finditer(r"LKW Multiplayer (?:product )?adoption complete", text):
        window = text[max(0, match.start() - 40) : match.end() + 80]
        assert re.search(r"(?i)\bnot\b|\bfalse\b", window), (
            f"unnegated adoption-complete claim: {window!r}"
        )


def test_required_certification_and_adr_links() -> None:
    text = _read(_MP_ARCH)
    missing = [m for m in _REQUIRED_LINKS if m not in text]
    assert not missing, f"missing certification/ADR refs: {missing}"


def test_satellite_docs_point_at_maturity_ssot() -> None:
    for path in (_ROADMAP, _ARCH_OVERVIEW, _PUBLIC_MAP, _CW_ARCH, _MP_PLAN):
        text = _read(path)
        assert "MULTIPLAYER_AI.md" in text, f"{path.name} must link Multiplayer architecture"
        assert (
            "enterprise core" in text.lower()
            or "ENTERPRISE CORE" in text
            or "Current Enterprise Maturity" in text
            or "ENTERPRISE BOUNDARY CERTIFIED" in text
        ), f"{path.name} missing enterprise-core maturity language"
