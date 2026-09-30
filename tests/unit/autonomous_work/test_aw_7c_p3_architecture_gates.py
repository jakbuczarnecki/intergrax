# © Artur Czarnecki. All rights reserved.

"""AW-7C-P3 architecture gates."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def test_orchestration_port_not_aliases_integrations_adaptation_port() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "intergrax/contracts/autonomous_work/scoped_adaptive_integration.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "WorkerScopedAdaptiveIntegrationOrchestrationPort = ScopedIntegrationAdaptationPort" not in text
    assert "class WorkerScopedAdaptiveIntegrationOrchestrationPort(Protocol):" in text
    assert "def prepare(" in text


def test_integrations_adaptation_port_is_adapt_only() -> None:
    path = (
        Path(__file__).resolve().parents[3]
        / "intergrax/integrations/contracts/scoped_integration_adaptation.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "class ScopedIntegrationAdaptationPort(Protocol):" in text
    assert "def adapt(" in text
