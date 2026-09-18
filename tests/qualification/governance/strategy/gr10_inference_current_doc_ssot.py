# © Artur Czarnecki. All rights reserved.

"""GR-10-R2-ADR1-R1 — current-status doc slices vs GR10_INFERENCE_CAPABILITY_SEMANTICS."""

from __future__ import annotations

import re
from pathlib import Path

from tests.qualification.governance.strategy.catalog import (
    GR10_INFERENCE_CAPABILITY_SEMANTICS,
    Gr10Applicability,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]

_ARCH = _REPO_ROOT / "docs" / "project" / "architecture" / "GOVERNED_EXECUTION.md"
_PLAN = _REPO_ROOT / "docs" / "project" / "maintainers" / "plans" / "GOVERNED_EXECUTION.md"
_GAP_LEDGER = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "GOVERNANCE_ARCHITECTURE_REBASE_GAP_LEDGER.md"
)
_QUAL = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "GOVERNANCE_FINAL_E2E_QUALIFICATION.md"
)

# False INFERENCE gap claims in *current* status sections (not historical tables).
_FORBIDDEN_INFERENCE_GAP_PHRASES: tuple[re.Pattern[str], ...] = (
    re.compile(r"INFERENCE\s+HITL/Reliability", re.IGNORECASE),
    re.compile(r"INFERENCE\s+HITL\s+residual", re.IGNORECASE),
    re.compile(r"HITL/Reliability\s+residual", re.IGNORECASE),
    re.compile(
        r"INFERENCE\s*/\s*AGENTIC\s+meaningful-side-effect\s+and\s+HITL",
        re.IGNORECASE,
    ),
)

_PRE_MODEL_INFERENCE_QUALIFIED = re.compile(
    r"PRE_MODEL.*(QUALIFIED|GR-10-R2-R1|GR-10-FINAL)",
    re.IGNORECASE | re.DOTALL,
)
_PRE_MODEL_ACTIVE_BLOCKER = re.compile(
    r"Active INFERENCE blocker:.*PRE_MODEL",
    re.IGNORECASE,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _section_after(text: str, heading: str, max_chars: int = 4000) -> str:
    start = text.find(heading)
    if start == -1:
        raise AssertionError(f"missing heading {heading!r}")
    chunk = text[start : start + max_chars]
    next_h2 = chunk.find("\n## ", len(heading))
    next_h3 = chunk.find("\n### ", len(heading))
    end_candidates = [c for c in (next_h2, next_h3) if c != -1]
    if end_candidates:
        return chunk[: min(end_candidates)]
    return chunk


def gr10_architecture_remaining_gaps_slice() -> str:
    text = _read(_ARCH)
    return _section_after(text, "### D. Remaining platform gaps (explicit)", 2500)


def gr10_maintainer_roadmap_slice() -> str:
    text = _read(_PLAN)
    return _section_after(text, "## GR enterprise roadmap (GOV-FINAL-1 code truth)", 3500)


def gr10_gap_ledger_gov_gap_008_slice() -> str:
    text = _read(_GAP_LEDGER)
    for line in text.splitlines():
        if line.startswith("| GOV-GAP-008 |"):
            return line
    raise AssertionError("GOV-GAP-008 row missing")


def gr10_qualification_current_status_slice() -> str:
    text = _read(_QUAL)
    section = _section_after(text, "### Current INFERENCE strategy semantics (GR-10-R1 SSOT)", 2000)
    return section


def gr10_forbidden_inference_gap_hits(slice_text: str) -> list[str]:
    return [pat.pattern for pat in _FORBIDDEN_INFERENCE_GAP_PHRASES if pat.search(slice_text)]


def gr10_assert_current_docs_inference_ssot() -> None:
    """Validate current-status governance docs against GR-10-R1 INFERENCE semantics."""
    slices = {
        "architecture_remaining_gaps": gr10_architecture_remaining_gaps_slice(),
        "maintainer_roadmap": gr10_maintainer_roadmap_slice(),
        "gov_gap_008": gr10_gap_ledger_gov_gap_008_slice(),
        "qualification_current_inference": gr10_qualification_current_status_slice(),
    }
    for name, body in slices.items():
        hits = gr10_forbidden_inference_gap_hits(body)
        assert not hits, f"{name} lists false INFERENCE gaps: {hits}"

    for name, body in (
        ("architecture_remaining_gaps", slices["architecture_remaining_gaps"]),
        ("maintainer_roadmap", slices["maintainer_roadmap"]),
        ("qualification_current_inference", slices["qualification_current_inference"]),
    ):
        assert _PRE_MODEL_INFERENCE_QUALIFIED.search(body), (
            f"{name} must record INFERENCE PRE_MODEL qualification (GR-10-FINAL / R2)"
        )
        assert not _PRE_MODEL_ACTIVE_BLOCKER.search(body), (
            f"{name} must not list PRE_MODEL as an active INFERENCE blocker"
        )

    na_caps = {
        row.capability
        for row in GR10_INFERENCE_CAPABILITY_SEMANTICS
        if row.applicability is Gr10Applicability.NOT_APPLICABLE
    }
    assert "HITL" in na_caps and "Reliability" in na_caps and "MSE" in na_caps
