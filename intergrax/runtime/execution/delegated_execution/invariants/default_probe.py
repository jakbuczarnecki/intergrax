# © Artur Czarnecki. All rights reserved.

"""Default delegated provider probe — neutral baseline for qualification."""

from __future__ import annotations

from intergrax.runtime.execution.delegated_execution.invariants.probe import (
    DelegatedProviderInvariantFacts,
)


class DefaultDelegatedProviderInvariantProbe:
    """No optional facts — rules evaluate as NOT_APPLICABLE or PASS defaults."""

    def read_facts(self) -> DelegatedProviderInvariantFacts:
        return DelegatedProviderInvariantFacts(
            provider_claims_canonical_execution_identity=False,
        )


__all__ = ["DefaultDelegatedProviderInvariantProbe"]
