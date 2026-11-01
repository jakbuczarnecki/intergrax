# © Artur Czarnecki. All rights reserved.

"""Render EBH_7_DUAL_SLUG_RPL_EF_MATRIX.md (F07)."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_DUAL_SLUG_RPL_EF_MATRIX.md"


def main() -> None:
    from tests.qualification.ebh_5._dual_slug_rpl_ef_matrix import render_dual_slug_rpl_ef_matrix_markdown
    from tests.qualification.ebh_7._ebh7_p03_evidence_integrity import _load_manifest

    sha = _load_manifest()["TEST_EXECUTION_SHA"]
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(render_dual_slug_rpl_ef_matrix_markdown(test_execution_sha=sha), encoding="utf-8")
    print(f"Wrote {_OUT.relative_to(_REPO)}")


if __name__ == "__main__":
    main()
