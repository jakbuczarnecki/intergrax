# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GUIDE = Path(__file__).resolve().parents[3] / "docs" / "guides/AGENT_CREATION_GUIDE.md"


def test_agent_creation_guide_documents_step_4e_scaffold() -> None:
    text = _GUIDE.read_text(encoding="utf-8")
    assert "### E — Dedicated application (scaffold)" in text
    assert "new-application" in text
    assert "new-stack" in text
    assert "--profile product" in text
    assert "build-docker.sh" in text
    assert "intergrax_runtime_architecture.md" in text
    assert "INTERGRAX_IMPLEMENTATION_PLAN.md" in text
    assert "poc_template_application" in text
