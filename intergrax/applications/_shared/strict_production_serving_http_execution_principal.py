# © Artur Czarnecki. All rights reserved.

"""Request-scoped HTTP caller principal for strict production serving execution (P3-A2)."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Iterator

from intergrax.contracts.agent_run import RequestIdentity

_strict_serving_http_caller_principal: ContextVar[RequestIdentity | None] = ContextVar(
    "strict_serving_http_caller_principal",
    default=None,
)


def get_strict_serving_http_caller_principal() -> RequestIdentity:
    principal = _strict_serving_http_caller_principal.get()
    if principal is None:
        raise ValueError(
            "strict production serving HTTP execution requires authenticated caller principal",
        )
    return principal


@contextmanager
def bind_strict_serving_http_caller_principal(
    principal: RequestIdentity,
) -> Iterator[None]:
    token: Token[RequestIdentity | None] = _strict_serving_http_caller_principal.set(principal)
    try:
        yield
    finally:
        _strict_serving_http_caller_principal.reset(token)


__all__ = [
    "bind_strict_serving_http_caller_principal",
    "get_strict_serving_http_caller_principal",
]
