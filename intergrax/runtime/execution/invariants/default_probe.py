# © Artur Czarnecki. All rights reserved.

"""Default execution invariant probe — contract constants + runtime module paths."""

from __future__ import annotations

from intergrax.contracts.execution_identity_authority import (
    CANONICAL_IDENTITY_AUTHORITY_MODULE,
    CANONICAL_LIFECYCLE_OWNER_MODULE,
)
from intergrax.runtime.execution.invariants.probe import ExecutionInvariantFacts


class DefaultExecutionInvariantProbe:
    """Production-safe read of canonical execution ownership anchors."""

    def read_facts(self) -> ExecutionInvariantFacts:
        return ExecutionInvariantFacts(
            identity_authority_module=CANONICAL_IDENTITY_AUTHORITY_MODULE,
            lifecycle_owner_module=CANONICAL_LIFECYCLE_OWNER_MODULE,
            supported_execution_bypass_active=False,
        )


__all__ = ["DefaultExecutionInvariantProbe"]
