# © Artur Czarnecki. All rights reserved.

"""Render final RPL-H convergence operator decision matrix (Phase C)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_RPL_FINAL_CONVERGENCE_DECISION_MATRIX.md"


def render() -> str:
    from tests.qualification.ebh_7._rpl_h_final_convergence_disposition import open_convergence_dispositions
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        all_normative_rpl_rows,
        normative_rpl_b6_open_concern_ids,
        normative_rpl_verdict_counts,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    open_ids = normative_rpl_b6_open_concern_ids()
    counts = normative_rpl_verdict_counts()
    lines = [
        "# EBH-7 — RPL-H final convergence decision matrix",
        "",
        "**Agent-prepared — not OD-RPL-G2 / OD-EBH7-SCOPE-RESIDUAL approval.**",
        "",
        f"**Normative counts:** QUALIFIED={counts[RplSemanticVerdict.QUALIFIED]} "
        f"PARTIAL={counts[RplSemanticVerdict.PARTIAL]} BLOCKED={counts[RplSemanticVerdict.BLOCKED]}",
        "",
        f"**B6 open concerns:** {len(open_ids)}",
        "",
        "## All 56 concerns — verified status @ HEAD",
        "",
        "| ID | SEMANTIC | VERIFIED | POSITIVE_PROOF |",
        "| --- | --- | --- | --- |",
    ]
    for row in all_normative_rpl_rows():
        lines.append(
            f"| {row.concern_id} | {row.semantic_verdict.value} | {row.verified_status.value} | "
            f"{row.positive_test[:100]} |"
        )
    lines.extend(
        [
            "",
            "## Open concerns — operator / vendor decisions required",
            "",
            "ID | ACTUAL_CONTRACT | CURRENT_EVIDENCE | MISSING_REQUIREMENT | RECOMMENDATION | "
            "OPTIONS | ARCHITECTURE_JUSTIFICATION | DECISION_REQUIRED | B6_IMPACT",
            "--- | --- | --- | --- | --- | --- | --- | --- | ---",
        ]
    )
    for d in open_convergence_dispositions():
        lines.append(
            f"| {d.concern_id} | {d.actual_contract.value} | {d.current_evidence} | "
            f"{d.missing_requirement} | {d.recommendation} | {d.options} | "
            f"{d.architecture_justification} | {'YES' if d.decision_required else 'NO'} | {d.b6_impact} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
