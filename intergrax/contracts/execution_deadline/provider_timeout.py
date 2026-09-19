# © Artur Czarnecki. All rights reserved.

"""Provider timeout derivation from execution deadline projection."""


def effective_provider_timeout_seconds(
    configured_timeout_seconds: float | None,
    remaining_execution_seconds: float | None,
) -> float | None:
    """
    Derive provider timeout from configured adapter timeout and remaining execution time.

    When no global deadline is configured, ``remaining_execution_seconds`` is ``None`` and the
    configured timeout is returned unchanged.
    """
    if remaining_execution_seconds is not None and remaining_execution_seconds <= 0:
        return 0.0
    if configured_timeout_seconds is None:
        if remaining_execution_seconds is None:
            return None
        return remaining_execution_seconds
    if remaining_execution_seconds is None:
        return configured_timeout_seconds
    return min(configured_timeout_seconds, remaining_execution_seconds)
