# © Artur Czarnecki. All rights reserved.

"""EE-B4-C — core runbooks remain provider-neutral."""

from __future__ import annotations

from pathlib import Path

import pytest

from testing_support.operations.incident_taxonomy import REQUIRED_RUNBOOK_IDS
from testing_support.operations.runbook_validator import split_runbook_blocks

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_RUNBOOKS_PATH = (
    _REPO
    / "docs"
    / "project"
    / "maintainers"
    / "runbooks"
    / "EXECUTION_ENGINE_PRODUCTION_RUNBOOKS.md"
)
_VENDOR_TOKENS = ("AWS", "Azure", "GCP", "OpenAI", "Anthropic")


def test_ee_b4_c_core_runbooks_exclude_vendor_specific_tokens() -> None:
    markdown = _RUNBOOKS_PATH.read_text(encoding="utf-8")
    blocks = split_runbook_blocks(markdown)
    for runbook_id in REQUIRED_RUNBOOK_IDS:
        block = blocks[runbook_id]
        for token in _VENDOR_TOKENS:
            assert token not in block.body, f"{runbook_id} mentions {token}"
