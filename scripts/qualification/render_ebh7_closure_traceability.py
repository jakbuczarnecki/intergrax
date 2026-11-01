# © Artur Czarnecki. All rights reserved.

"""Generate closure traceability stub linking packages to gates."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_FINAL_CLOSURE_TRACEABILITY.md"


def render() -> str:
    import subprocess

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()
    lines = [
        "# EBH-7 — Final closure traceability",
        "",
        f"**HEAD @ generation:** `{head}`",
        "",
        "| Package | Requirement | Gate / test | SHA role |",
        "| --- | --- | --- | --- |",
        "| A | SSOT RPL/TEN/XDM | `_ebh7_certification_ssot.py` | FINAL_IMPLEMENTATION |",
        "| B | INT2E bundles | `test_ebh_7_int2e_bundle_regression.py` | TEST_EXECUTION |",
        "| C | RPL-H normative | `assert_ebh7_rpl_h_normative_for_b6_readiness` | B6 prep |",
        "| D | XDM integrated | `test_ebh_7_certification_ssot_gates` | TEST_EXECUTION |",
        "| E | ECAP C1–C5 | `EBH_7_ECAP_EXECUTABLE_CHECKLIST.md` | EVIDENCE_PUBLICATION |",
        "| F | L3 consolidated | full `tests/qualification/ebh_7/` | TEST_EXECUTION_SHA |",
        "| G | P03 B1–B5 | `ebh_7_p03_independent_replay/MANIFEST.json` | EVIDENCE_PUBLICATION_SHA |",
        "| H | B6 independent | `EBH_7_PRE_CERT_INDEPENDENT_AUDIT_WORKSHEET.md` | INDEPENDENT_ACCEPTANCE |",
        "| I | OD-EBH7-08 | closure signature record | FORMAL_CLOSURE |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
