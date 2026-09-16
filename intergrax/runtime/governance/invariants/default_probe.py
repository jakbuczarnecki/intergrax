# © Artur Czarnecki. All rights reserved.

from intergrax.runtime.governance.invariants.probe import GovernanceInvariantFacts


class DefaultGovernanceInvariantProbe:
    def read_facts(self) -> GovernanceInvariantFacts:
        return GovernanceInvariantFacts()


__all__ = ["DefaultGovernanceInvariantProbe"]
