# © Artur Czarnecki. All rights reserved.

"""Sub-engine protocol for adaptation recommend wave (Phase W-ADAPT-2)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationEngineContext,
    AdaptationProposalCandidate,
)


class AdaptationSubEngine(Protocol):
    """Produces bounded proposal candidates from signal history."""

    @property
    def engine_id(self) -> str: ...

    def propose(self, context: AdaptationEngineContext) -> list[AdaptationProposalCandidate]: ...
