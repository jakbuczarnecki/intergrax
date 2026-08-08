# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SIGNOFF = REPO_ROOT / "agents" / "signoff_probe"

REQUIRED = (
    "signoff_probe_agent.py",
    "contract.py",
    "capabilities.py",
    "tests/test_signoff_probe_agent.py",
    "docs/ARCHITECTURE.md",
    "docs/IMPLEMENTATION_PLAN.md",
    "docs/project/technical/adr/README.md",
    "docs/project/technical/adr/TEMPLATE.md",
)


@pytest.mark.gate
def test_signoff_probe_matches_scaffold_layout() -> None:
    for rel in REQUIRED:
        assert (SIGNOFF / rel).is_file(), f"missing {rel}"
