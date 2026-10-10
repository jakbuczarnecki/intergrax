# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve observability backends by harness role (errors vs traces vs default)."""

from __future__ import annotations

from intergrax.integrations.contracts.observability_backend import ObservabilityBackend
from intergrax.tools.registry.wiring import ToolWiringContext

_KNOWN_ROLES = frozenset({"errors", "traces", "logs", "eval"})


def resolve_observability_backend(
    ctx: ToolWiringContext,
    *,
    role: str = "default",
) -> ObservabilityBackend:
    """
    Pick an observability backend for a tool capability.

    Role backends are materialized only from ``IntegrationProfile.observability_roles``.
    """
    if role == "default":
        if ctx.observability_backend is None:
            raise RuntimeError("observability_backend_not_configured")
        return ctx.observability_backend

    if role not in _KNOWN_ROLES:
        raise RuntimeError(f"observability_unknown_role:{role}")

    role_backends = ctx.observability_role_backends
    backend: ObservabilityBackend | None
    if role == "errors":
        backend = role_backends.errors
    elif role == "traces":
        backend = role_backends.traces
    elif role == "logs":
        backend = role_backends.logs
    else:
        backend = role_backends.eval

    if backend is None:
        raise RuntimeError(f"observability_role_backend_not_configured:{role}")
    return backend
