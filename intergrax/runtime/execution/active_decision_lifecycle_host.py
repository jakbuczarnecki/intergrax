# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution-scoped active Decision Lifecycle host binding (DS-EXEC-01)."""

from __future__ import annotations

from contextvars import ContextVar, Token

from intergrax.runtime.execution.decision_lifecycle_host import DecisionLifecycleHost

_active_decision_lifecycle_host: ContextVar[DecisionLifecycleHost | None] = ContextVar(
    "active_decision_lifecycle_host",
    default=None,
)


def bind_active_decision_lifecycle_host(host: DecisionLifecycleHost) -> Token:
    return _active_decision_lifecycle_host.set(host)


def reset_active_decision_lifecycle_host(token: Token) -> None:
    _active_decision_lifecycle_host.reset(token)


def get_active_decision_lifecycle_host() -> DecisionLifecycleHost | None:
    return _active_decision_lifecycle_host.get()


def require_active_decision_lifecycle_host() -> DecisionLifecycleHost:
    host = get_active_decision_lifecycle_host()
    if host is None:
        raise RuntimeError("active decision lifecycle host required")
    return host
