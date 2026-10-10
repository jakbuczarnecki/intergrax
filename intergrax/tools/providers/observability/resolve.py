# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve observability backends by harness role (errors vs traces vs default)."""

from __future__ import annotations
from intergrax.utils import attribute_access

from typing import Any

from intergrax.tools.registry.wiring import ToolWiringContext

_ERRORS_SLUGS = ("sentry",)
_TRACES_SLUGS = ("langsmith", "langfuse", "braintrust", "phoenix", "signoz", "helicone")
_EVAL_SLUGS = ("braintrust",)
_LOGS_SLUGS = ("elasticsearch", "opensearch")


def _backends(ctx: ToolWiringContext) -> dict[str, Any]:
    if ctx.observability_backends:
        return ctx.observability_backends
    if ctx.observability_backend is not None:
        return {"default": ctx.observability_backend}
    return {}


def _sanctioned_slug_backend(
    backends: dict[str, Any],
    slug_order: tuple[str, ...],
    *,
    attr: str,
) -> Any | None:
    for slug in slug_order:
        candidate = backends.get(slug)
        if candidate is not None and attribute_access.optional(candidate, attr, None) is not None:
            return candidate
    return None


def _raise_role_not_configured(role: str) -> None:
    raise RuntimeError(f"observability_role_backend_not_configured:{role}")


def resolve_observability_backend(ctx: ToolWiringContext, *, role: str = "default") -> Any:
    """
    Pick an observability backend for a tool capability.

    Roles:
    - ``errors`` — Sentry-like ``capture_message`` (sanctioned slugs only)
    - ``traces`` — ``query_traces`` (sanctioned slugs only)
    - ``logs`` — ``rest_client`` for log search (sanctioned slugs only)
    - ``eval`` — ``log_eval`` (sanctioned slugs only)
    - ``default`` — explicit ``observability_backend`` only (no registration-order fallback)
    """
    backends = _backends(ctx)
    if not backends and ctx.observability_backend is None:
        raise RuntimeError("observability_backend_not_configured")

    if role == "errors":
        backend = _sanctioned_slug_backend(backends, _ERRORS_SLUGS, attr="capture_message")
        if backend is not None:
            return backend
        _raise_role_not_configured(role)
    if role == "traces":
        backend = _sanctioned_slug_backend(backends, _TRACES_SLUGS, attr="query_traces")
        if backend is not None:
            return backend
        _raise_role_not_configured(role)
    if role == "logs":
        backend = _sanctioned_slug_backend(backends, _LOGS_SLUGS, attr="rest_client")
        if backend is not None:
            return backend
        _raise_role_not_configured(role)
    if role == "eval":
        backend = _sanctioned_slug_backend(backends, _EVAL_SLUGS, attr="log_eval")
        if backend is not None:
            return backend
        _raise_role_not_configured(role)

    if ctx.observability_backend is not None:
        return ctx.observability_backend
    raise RuntimeError("observability_backend_not_configured")
