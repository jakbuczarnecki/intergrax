# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Standalone FastAPI app for trace inspection and experiment registry (Phase D.2–D.3)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI

from intergrax.debug.router import create_debug_router
from intergrax.fastapi_core.routers.health import health_router


def create_debug_app(
    *,
    db_path: Path | None = None,
    experiments_db_path: Path | None = None,
) -> FastAPI:
    """
    Laboratory debug API over SQLite trace store and experiment registry.

    Usage::

        uvicorn intergrax.debug.app:create_debug_app --factory --reload
    """
    app = FastAPI(
        title="Intergrax Debug API",
        version="0.1.0",
        description="Inspect Nexus runs/traces and manage experiments (Phase D, §19, §35).",
    )
    app.include_router(health_router)
    app.include_router(
        create_debug_router(
            db_path=db_path,
            experiments_db_path=experiments_db_path,
        )
    )
    return app
