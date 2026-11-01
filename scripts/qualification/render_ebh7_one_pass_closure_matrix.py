# © Artur Czarnecki. All rights reserved.

"""Render EBH-7 one-pass closure matrix."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_ONE_PASS_CLOSURE_MATRIX.md"


def render() -> str:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        all_normative_rpl_rows,
        normative_rpl_b6_open_concern_ids,
        normative_rpl_verdict_counts,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    open_ids = normative_rpl_b6_open_concern_ids()
    counts = normative_rpl_verdict_counts()
    lines = [
        "# EBH-7 — one-pass closure matrix",
        "",
        f"**QUALIFIED:** {counts[RplSemanticVerdict.QUALIFIED]}/56 · "
        f"**PARTIAL:** {counts[RplSemanticVerdict.PARTIAL]} · **BLOCKED:** {counts[RplSemanticVerdict.BLOCKED]}",
        "",
        f"**B6 open:** {len(open_ids)}",
        "",
        "| REQUIREMENT (concern) | TASK | VERIFIED IMPLEMENTATION | EXACT EVIDENCE | TRUE STATUS | REMAINING GATE |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in all_normative_rpl_rows():
        gate = "—"
        if row.concern_id in open_ids:
            gate = "OD-RPL-G2 / architecture" if row.semantic_verdict is RplSemanticVerdict.BLOCKED else "OD or qualify"
        lines.append(
            f"| {row.concern_id} | RPL-H | normative SSOT | {row.positive_test[:80]} | "
            f"{row.semantic_verdict.value} | {gate} |"
        )
    lines.extend(
        [
            "",
            "**Program verdict:** `EBH-7 — TECHNICAL WORK COMPLETE / EXACT EXTERNAL AUTHORIZATIONS REQUIRED` "
            "when B6 open > 0 without scoped OD.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
