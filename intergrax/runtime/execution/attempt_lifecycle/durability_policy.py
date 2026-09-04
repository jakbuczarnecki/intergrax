# © Artur Czarnecki. All rights reserved.

"""Production durability requirements for attempt lifecycle retry transitions (P0C-4A)."""

from __future__ import annotations

from intergrax.contracts.attempt_lifecycle import AttemptLifecycleError, AttemptLifecycleStore

DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG = (
    "durable attempt lifecycle store required for retry execution"
)


def retry_transition_capability_requested(
    *,
    agent_retry_max: int,
    run_retry_max: int,
) -> bool:
    """Return whether configured retry policy may create a new Attempt."""
    return agent_retry_max > 0 or run_retry_max > 0


def validate_durable_attempt_lifecycle_for_composition(
    *,
    production_mode: bool,
    store: AttemptLifecycleStore,
    agent_retry_max: int,
    run_retry_max: int,
) -> None:
    """Fail closed at composition when production retry requires durable authority."""
    if not production_mode:
        return
    if not retry_transition_capability_requested(
        agent_retry_max=agent_retry_max,
        run_retry_max=run_retry_max,
    ):
        return
    if not store.is_durable:
        raise AttemptLifecycleError(DURABLE_ATTEMPT_LIFECYCLE_REQUIRED_MSG)
