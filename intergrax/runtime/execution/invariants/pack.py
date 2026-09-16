# © Artur Czarnecki. All rights reserved.

"""Execution engine runtime invariant rule pack."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.runtime_invariants import RuntimeInvariantDomain, RuntimeInvariantRule
from intergrax.runtime.execution.invariants.probe import ExecutionInvariantProbe
from intergrax.runtime.execution.invariants.rules import (
    ExecutionCanonicalIdentityAuthorityRule,
    ExecutionCanonicalLifecycleOwnerRule,
    ExecutionNoSupportedBypassRule,
)

_PACK_ID = "execution-foundation"
_PACK_VERSION = "1.0.0"


@dataclass(frozen=True, slots=True)
class ExecutionRuntimeInvariantRulePack:
    """Domain-owned execution invariant pack."""

    probe: ExecutionInvariantProbe
    pack_id: str = _PACK_ID
    pack_version: str = _PACK_VERSION
    domain: RuntimeInvariantDomain = RuntimeInvariantDomain.EXECUTION

    @property
    def rules(self) -> tuple[RuntimeInvariantRule, ...]:
        return (
            ExecutionCanonicalIdentityAuthorityRule(self.probe),
            ExecutionCanonicalLifecycleOwnerRule(self.probe),
            ExecutionNoSupportedBypassRule(self.probe),
        )


__all__ = ["ExecutionRuntimeInvariantRulePack"]
