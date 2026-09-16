# © Artur Czarnecki. All rights reserved.

"""Delegated provider runtime invariant rule pack."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_invariants import RuntimeInvariantDomain, RuntimeInvariantDomains, RuntimeInvariantRule
from intergrax.runtime.execution.delegated_execution.invariants.probe import (
    DelegatedProviderInvariantProbe,
)
from intergrax.runtime.execution.delegated_execution.invariants.rules import (
    DelegatedCorrelationImmutableRule,
    DelegatedNoCorrelationFallbackSynthesisRule,
    DelegatedProviderNoCanonicalIdentityOwnershipRule,
    DelegatedReattachNoNewWorkRule,
)

_PACK_ID = "delegated-provider-foundation"
_PACK_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class DelegatedProviderRuntimeInvariantRulePack:
    probe: DelegatedProviderInvariantProbe
    pack_id: str = _PACK_ID
    pack_version: str = _PACK_VERSION
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.DELEGATED_PROVIDER

    @property
    def rules(self) -> tuple[RuntimeInvariantRule, ...]:
        return (
            DelegatedProviderNoCanonicalIdentityOwnershipRule(self.probe),
            DelegatedCorrelationImmutableRule(self.probe),
            DelegatedNoCorrelationFallbackSynthesisRule(self.probe),
            DelegatedReattachNoNewWorkRule(self.probe),
        )


__all__ = ["DelegatedProviderRuntimeInvariantRulePack"]
