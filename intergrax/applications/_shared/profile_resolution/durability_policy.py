# © Artur Czarnecki. All rights reserved.

"""Production durability requirements for effective profile revision pinning (P1.2A)."""

from __future__ import annotations

from intergrax.applications.contracts.profile_resolution.errors import EffectiveProfileRevisionError

DURABLE_EFFECTIVE_PROFILE_REVISION_REQUIRED_MSG = (
    "durable effective profile revision store required for production execution pinning"
)
DURABLE_EFFECTIVE_PROFILE_PINNING_REQUIRED_MSG = (
    "durable effective profile execution pinning store required for production execution"
)


def _is_durable_store(store: object) -> bool:
    durable = getattr(store, "is_durable", None)
    return bool(durable) if durable is not None else False


def validate_effective_profile_pinning_durability_for_composition(
    *,
    production_mode: bool,
    revision_store: object,
    pinning_store: object,
) -> None:
    """Fail closed when production host composition lacks durable revision authorities."""
    if not production_mode:
        return
    if not _is_durable_store(revision_store):
        raise EffectiveProfileRevisionError(DURABLE_EFFECTIVE_PROFILE_REVISION_REQUIRED_MSG)
    if not _is_durable_store(pinning_store):
        raise EffectiveProfileRevisionError(DURABLE_EFFECTIVE_PROFILE_PINNING_REQUIRED_MSG)
