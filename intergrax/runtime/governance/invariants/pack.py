# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_invariants import RuntimeInvariantDomain, RuntimeInvariantDomains, RuntimeInvariantRule
from intergrax.runtime.governance.invariants.probe import GovernanceInvariantProbe
from intergrax.runtime.governance.invariants.rules import GovernanceInnerExecutionBindingRule

_PACK_ID = "governance-foundation"
_PACK_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class GovernanceRuntimeInvariantRulePack:
    probe: GovernanceInvariantProbe
    pack_id: str = _PACK_ID
    pack_version: str = _PACK_VERSION
    domain: RuntimeInvariantDomain = RuntimeInvariantDomains.GOVERNANCE

    @property
    def rules(self) -> tuple[RuntimeInvariantRule, ...]:
        return (GovernanceInnerExecutionBindingRule(self.probe),)


__all__ = ["GovernanceRuntimeInvariantRulePack"]
