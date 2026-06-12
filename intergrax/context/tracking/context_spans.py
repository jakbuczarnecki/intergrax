# © Artur Czarnecki. All rights reserved.

"""OTel span names for context engineering (CE-9.2)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

CE_OTEL_SPAN_NAMES: tuple[str, ...] = (
    "context.engine.assemble",
    "context.provider.collect",
    "context.budget.allocate",
)


@contextmanager
def context_span(name: str) -> Iterator[None]:
    """No-op span shim when OTel SDK is not configured."""
    if name not in CE_OTEL_SPAN_NAMES:
        raise ValueError(f"unknown context span: {name}")
    yield
