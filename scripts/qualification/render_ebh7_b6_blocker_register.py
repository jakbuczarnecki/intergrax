# © Artur Czarnecki. All rights reserved.

"""Render B6 blocker register and operator decision summary."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_B6_BLOCKER_REGISTER.md"


def render() -> str:
    from tests.qualification.ebh_7._rpl_h_normative_evidence import (
        all_normative_rpl_rows,
        normative_rpl_b6_open_concern_ids,
    )
    from tests.qualification.ebh_7._rpl_h_semantic_evidence import RplSemanticVerdict

    open_ids = normative_rpl_b6_open_concern_ids()
    lines = [
        "# EBH-7 — B6 blocker register @ HEAD",
        "",
        f"**Open RPL concerns (semantic not QUALIFIED):** {len(open_ids)}",
        "",
        "| ID | Semantic | Verified | Defect | Operator path |",
        "| --- | --- | --- | --- | --- |",
    ]
    by_id = {r.concern_id: r for r in all_normative_rpl_rows()}
    for cid in open_ids:
        row = by_id[cid]
        od = "OD-RPL-G2" if row.semantic_verdict is RplSemanticVerdict.BLOCKED else "OD-SCOPE / qualify"
        lines.append(
            f"| {cid} | {row.semantic_verdict.value} | {row.verified_status.value} | {row.defect[:80]} | {od} |"
        )
    lines.extend(
        [
            "",
            "**TEN/XDM:** disposition EXACT + evidence SSOT (0 COMPOSED_PROXY); not a substitute for RPL B6 PASS.",
            "",
            "**Not approved:** B6 PASS, OD-RPL-G2, OD-EBH7-SCOPE-RESIDUAL, OD-EBH7-08.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
