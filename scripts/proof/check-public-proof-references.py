#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Validate explicit public documentation proof references (PUBLIC-PROOF-GATE-2)."""

from __future__ import annotations

import sys
from pathlib import Path

from scripts.proof.public_proof_references import (
    render_report,
    validate_public_proof_references,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    report = validate_public_proof_references(repo_root=_REPO_ROOT)
    print(render_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
