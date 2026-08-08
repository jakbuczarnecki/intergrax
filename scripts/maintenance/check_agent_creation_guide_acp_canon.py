#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""ACP-CLOSE-LEG-4 + PAT-3 — author guide ACP canon and §29 terminology entry."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUIDE = REPO_ROOT / "docs" / "project" / "technical" / "guides" / "AGENT_CREATION_GUIDE.md"

# Author-facing phrases that imply UAEP as the primary implementation path.
FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "UAEP-first",
    "Bridge (today)",
    "until `ACP-STEP-3`",
    "UAEP entry point",
    "UAEP steps implemented",
    "via UAEP — do not open",
    "during UAEP.",
    "Inside `run_step`:",
    "New agents on legacy UAEP only after Wave 5",
    "phased `get_steps`",
    "+ UAEP steps",
    "Target (ACP — after Wave 5)",
    "until then use RuntimeRequest",
)

REQUIRED_MARKERS: tuple[str, ...] = (
    "UAEP is harness-internal only",
    "on_next_step",
    "ACP-CLOSE-LEG-4",
    "Author terminology canon",
    "#29-author-facing-run-facade",
    "ACP-CLOSE-PAT-3",
)


def main() -> int:
    if not GUIDE.is_file():
        print(f"Missing guide: {GUIDE}")
        return 1

    text = GUIDE.read_text(encoding="utf-8")
    violations: list[str] = []

    for needle in FORBIDDEN_SUBSTRINGS:
        if needle in text:
            violations.append(f"forbidden author-path phrase: {needle!r}")

    for marker in REQUIRED_MARKERS:
        if marker not in text:
            violations.append(f"missing required canonical marker: {marker!r}")

    if violations:
        print("AGENT_CREATION_GUIDE ACP canon violations:")
        print("\n".join(violations))
        return 1

    print("AGENT_CREATION_GUIDE ACP canon: OK (ACP-CLOSE-LEG-4 · PAT-3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
