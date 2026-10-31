# © Artur Czarnecki. All rights reserved.

"""Generate consolidated RPL operator decision package from normative SSOT."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_RPL_OPERATOR_DECISION_PACKAGE.md"


def _recommended_action(concern_id: str, status: str) -> tuple[str, str, str]:
    if status == "BLOCKED_ARCHITECTURE":
        return (
            "OD-RPL-G2 fixed substrate ADR or fund second provider sandbox",
            "ALLOWED_FIXED; second provider; defer B6",
            "YES",
        )
    if status == "DUAL_SLUG_REGISTRY_ONLY":
        return (
            "Extend RPL-C/D/E equivalence proof or scoped residual",
            "Behavioral equivalence test; OD-SCOPE-RESIDUAL; demote matrix",
            "YES",
        )
    if status == "FACTORY_ONLY":
        return (
            "Distinct alternate/negative proof or fixed-substrate classification",
            "Package C negative path; ALLOWED_FIXED; defer",
            "YES",
        )
    return ("Maintain demonstrated evidence", "N/A", "NO")


def render_rpl_operator_decision_package() -> str:
    from tests.qualification.ebh_5._replaceability_matrix import CROSS_LAYER_REPLACEABILITY_MATRIX
    from tests.qualification.ebh_7._rpl_h_normative_evidence import all_normative_rpl_rows

    matrix = {r.concern_id: r for r in CROSS_LAYER_REPLACEABILITY_MATRIX}
    lines = [
        "# EBH-7 — RPL operator decision package (consolidated)",
        "",
        "Agent-prepared — **not** OD-RPL-G2 / OD-EBH7-SCOPE-RESIDUAL approval.",
        "",
        "| ID | CANONICAL_OWNER | CLAIMED_EXTENSIBILITY | VERIFIED_PROOF | EXACT_GAP | "
        "RECOMMENDED_ACTION | OPTIONS | OPERATOR_DECISION_REQUIRED | B6_IMPACT |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in all_normative_rpl_rows():
        m = matrix[row.concern_id]
        if row.verified_status.value == "DEMONSTRATED":
            continue
        rec, opts, od = _recommended_action(row.concern_id, row.verified_status.value)
        owner = m.concern_id.split(".")[0] if "." in m.concern_id else "platform"
        if m.concern_id.startswith("integration."):
            owner = "intergrax/integrations"
        elif m.concern_id.startswith("runtime."):
            owner = "intergrax/runtime"
        ext = str(m.rpl_h).replace("|", " ")
        lines.append(
            f"| {row.concern_id} | {owner} | {ext} | {row.positive_test} | {row.defect} | "
            f"{rec} | {opts} | {od} | Blocks normative B6 until resolved or scoped |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render_rpl_operator_decision_package(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
