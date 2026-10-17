# © Artur Czarnecki. All rights reserved.

"""ASGI lifespan ownership for strict production process lifecycle (PROD-Q-R1-R7-D4)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI

from intergrax.applications._shared.fastapi_lifespan import LifespanFn, apply_lifespans
from intergrax.applications._shared.production_process_lifecycle import (
    ProductionProcessLifecycle,
)
from intergrax.applications._shared.strict_production_serving_launcher import (
    StrictProductionServingSession,
)


def make_strict_production_process_lifespan(
    lifecycle: ProductionProcessLifecycle,
) -> LifespanFn:
    """Close production-owned durable stores when the ASGI server shuts down."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            lifecycle.close_process_resources()

    return cast(LifespanFn, _lifespan)


def attach_strict_production_serving_lifespan(
    app: FastAPI,
    session: StrictProductionServingSession,
) -> FastAPI:
    """Bind ASGI shutdown to the strict production serving session lifecycle owner."""

    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            if not session._closed and not session.lifecycle._process_closed:
                session.close()

    return apply_lifespans(app, cast(LifespanFn, _lifespan))


__all__ = [
    "attach_strict_production_serving_lifespan",
    "make_strict_production_process_lifespan",
]
