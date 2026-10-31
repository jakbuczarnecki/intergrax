# © Artur Czarnecki. All rights reserved.

"""Render EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md from closure audit SSOT."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    from tests.qualification.ebh_7._ebh7_closure_audit_markdown_sync import write_closure_evidence_audit_markdown

    write_closure_evidence_audit_markdown()
    rel = "docs/project/maintainers/qualification/ebh_7_p03_independent_replay/EBH_7_CLOSURE_EVIDENCE_AUDIT_MATRIX.md"
    print(f"Wrote {rel}")


if __name__ == "__main__":
    main()
