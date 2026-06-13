# © Artur Czarnecki. All rights reserved.

"""APP-CON-DX.1 — application creation guide."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]

GUIDE = Path(__file__).resolve().parents[3] / "docs" / "guides" / "APPLICATION_CREATION_GUIDE.md"


def test_application_creation_guide_exists_with_canon_sections() -> None:
    text = GUIDE.read_text(encoding="utf-8")
    assert GUIDE.is_file()
    for marker in (
        "## 1. Mental model",
        "## 2. Author workflow",
        "## 3. New application checklist",
        "build_harness_host_runtime",
        "intergrax apps list",
    ):
        assert marker in text
