# © Artur Czarnecki. All rights reserved.

"""Architecture gate: production Legal serving must not assemble AgentRegistry locally."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO = Path(__file__).resolve().parents[4]
LEGAL_FASTAPI_ROUTER = (
    REPO / "applications" / "legal_application" / "serving" / "fastapi_router.py"
)


def test_legal_serving_does_not_bootstrap_agent_registry_locally() -> None:
    source = LEGAL_FASTAPI_ROUTER.read_text(encoding="utf-8")
    forbidden = (
        "AgentRegistry.from_agents",
        "AgentRegistry()",
        ".from_agents(",
        "agents: Optional",
        "agents must be provided",
    )
    for token in forbidden:
        assert token not in source, (
            f"legal production serving must consume canonical registry projection; "
            f"found forbidden token {token!r}"
        )
