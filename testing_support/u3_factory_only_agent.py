# © Artur Czarnecki. All rights reserved.

"""Test-only agent requiring factory mount (U3 governance materialization)."""

from __future__ import annotations

from echo.echo_agent import EchoAgent
from intergrax.contracts.agent_contract_meta import AgentContract

_FACTORY_CAPABILITY = "factory-only.run"


class FactoryOnlyAgent(EchoAgent):
    """Not constructible via zero-argument constructor."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("token required")
        super().__init__()

    def get_contract(self) -> AgentContract:
        return super().get_contract().model_copy(
            update={"id": "factory-only", "capabilities": [_FACTORY_CAPABILITY]},
        )


def build_factory_only_agent(*_args: object, **_kwargs: object) -> FactoryOnlyAgent:
    return FactoryOnlyAgent("mounted")
