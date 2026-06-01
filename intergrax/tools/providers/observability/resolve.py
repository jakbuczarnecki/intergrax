# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Resolve observability backends by harness role (errors vs traces vs default)."""

from __future__ import annotations

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


def _first_matching(
    backends: dict[str, Any],
    slug_order: tuple[str, ...],
    *,
    attr: str,
) -> Any | None:
    for slug in slug_order:
        candidate = backends.get(slug)
        if candidate is not None and getattr(candidate, attr, None) is not None:
            return candidate
    for candidate in backends.values():
        if getattr(candidate, attr, None) is not None:
            return candidate
    return None


def resolve_observability_backend(ctx: ToolWiringContext, *, role: str = "default") -> Any:
    """
    Pick an observability backend for a tool capability.

    Roles:
    - ``errors`` — Sentry-like ``capture_message`` (prefers ``sentry`` slug)
    - ``traces`` — ``query_traces`` (prefers ``langsmith``, ``langfuse``, …)
    - ``logs`` — ``rest_client`` for log search (prefers elasticsearch/opensearch)
    - ``default`` — primary ``observability_backend`` or first registered backend
    """
    backends = _backends(ctx)
    if role == "errors":
        backend = _first_matching(backends, _ERRORS_SLUGS, attr="capture_message")
        if backend is not None:
            return backend
    if role == "traces":
        backend = _first_matching(backends, _TRACES_SLUGS, attr="query_traces")
        if backend is not None:
            return backend
    if role == "logs":
        backend = _first_matching(backends, _LOGS_SLUGS, attr="rest_client")
        if backend is not None:
            return backend
        for candidate in backends.values():
            if getattr(candidate, "rest_client", None) is not None:
                return candidate
    if role == "eval":
        backend = _first_matching(backends, _EVAL_SLUGS, attr="log_eval")
        if backend is not None:
            return backend

    if ctx.observability_backend is not None:
        return ctx.observability_backend
    if backends:
        return next(iter(backends.values()))
    raise RuntimeError("observability_backend_not_configured")
