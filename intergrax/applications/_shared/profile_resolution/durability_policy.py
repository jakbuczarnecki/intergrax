# © Artur Czarnecki. All rights reserved.

"""Production durability requirements for effective profile revision pinning (P1.2A)."""

from __future__ import annotations

from typing import Protocol

from intergrax.applications.contracts.profile_resolution.errors import EffectiveProfileRevisionError

DURABLE_EFFECTIVE_PROFILE_REVISION_REQUIRED_MSG = (
    "durable effective profile revision store required for production execution pinning"
)
DURABLE_EFFECTIVE_PROFILE_PINNING_REQUIRED_MSG = (
    "durable effective profile execution pinning store required for production execution"
)


class DurablePersistenceCapability(Protocol):
    @property
    def is_durable(self) -> bool:
        """Whether state survives process restart."""


def validate_effective_profile_pinning_durability_for_composition(
    *,
    production_mode: bool,
    revision_store: DurablePersistenceCapability,
    pinning_store: DurablePersistenceCapability,
) -> None:
    """Fail closed when production host composition lacks durable revision authorities."""
    if not production_mode:
        return
    if not revision_store.is_durable:
        raise EffectiveProfileRevisionError(DURABLE_EFFECTIVE_PROFILE_REVISION_REQUIRED_MSG)
    if not pinning_store.is_durable:
        raise EffectiveProfileRevisionError(DURABLE_EFFECTIVE_PROFILE_PINNING_REQUIRED_MSG)
