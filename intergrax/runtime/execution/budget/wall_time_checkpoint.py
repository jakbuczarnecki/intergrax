# © Artur Czarnecki. All rights reserved.

"""Wall-time delta checkpoint for canonical budget consumption (UE-8B2)."""

from __future__ import annotations

from contextvars import ContextVar, Token

_last_wall_time_accounted: ContextVar[float] = ContextVar(
    "execution_budget_wall_time_accounted",
    default=0.0,
)


def reset_wall_time_accounting() -> None:
    """Reset wall-time delta checkpoint for the active execution budget scope."""
    _last_wall_time_accounted.set(0.0)


def bind_wall_time_accounting_checkpoint(elapsed_seconds: float) -> Token[float]:
    """Bind an explicit wall-time checkpoint (used when callers already know elapsed)."""
    return _last_wall_time_accounted.set(elapsed_seconds)


def reset_wall_time_accounting_token(token: Token[float]) -> None:
    _last_wall_time_accounted.reset(token)


def wall_time_delta_since_checkpoint(elapsed_seconds: float) -> float:
    """Return positive elapsed delta since the last checkpoint."""
    last_accounted = _last_wall_time_accounted.get()
    delta = elapsed_seconds - last_accounted
    if delta <= 0.0:
        return 0.0
    _last_wall_time_accounted.set(elapsed_seconds)
    return delta
