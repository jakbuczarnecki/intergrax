"""Proof/test control surface for the controlled order provider — not business API."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.indirect_prompt_injection.application.order_provider_models import (
    OrderProviderNote,
    ProviderMutationState,
)


class OrderProviderControlPort(Protocol):
    """Reset and observe provider mutations; consumed by proof harness only."""

    def reset(self, *, notes: list[OrderProviderNote] | None = None) -> None: ...

    def mutation_state(self) -> ProviderMutationState: ...
