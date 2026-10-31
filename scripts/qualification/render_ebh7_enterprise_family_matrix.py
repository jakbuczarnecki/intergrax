# © Artur Czarnecki. All rights reserved.

"""Expand enterprise family coverage from EBH-6 family catalog."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/EBH_7_ENTERPRISE_FAMILY_COVERAGE_MATRIX.md"


def render() -> str:
    from tests.qualification.ebh_6._family_catalog import EBH6_FAMILY_CATALOG

    lines = [
        "# EBH-7 — Enterprise architecture family coverage (closure)",
        "",
        "| Family ID | Title | Owner gate | EBH-7 task | Evidence @ HEAD |",
        "| --- | --- | --- | --- | --- |",
    ]
    for fam in EBH6_FAMILY_CATALOG:
        gate = fam.gate_test_ids[0] if fam.gate_test_ids else "mechanical"
        lines.append(
            f"| {fam.family_id} | {fam.ecap_theme} | {gate} | EBH7-FC-E-{fam.family_id} | "
            f"EBH-6 catalog + ebh_7 replay @ HEAD |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    _OUT.write_text(render(), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
