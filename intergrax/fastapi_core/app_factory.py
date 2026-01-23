# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import FastAPI

from intergrax.fastapi_core.auth.provider import AuthProvider
from intergrax.fastapi_core.auth.providers.api_key.provider import ApiKeyAuthProvider
from intergrax.fastapi_core.auth.providers.compose.provider import CompositeAuthProvider
from intergrax.fastapi_core.auth.providers.no_auth import NoAuthProvider
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.errors.handlers import global_exception_handler
from intergrax.fastapi_core.middleware.request_context import RequestContextMiddleware
from intergrax.fastapi_core.rate_limit.policy import RateLimitPolicy
from intergrax.fastapi_core.routers.health import health_router
from intergrax.fastapi_core.runs.router import runs_router
from intergrax.fastapi_core.runs.store_base import RunStore
from intergrax.fastapi_core.runs.store_memory import InMemoryRunStore


def create_app(config: ApiConfig) -> FastAPI:
    """
    Application factory for Intergrax FastAPI Core.

    Notes:
    - Must remain side-effect free (no implicit global initialization).
    - All dependencies are wired explicitly (DI style).
    """
    config.validate()

    is_prod: bool = config.environment == ApiEnvironment.PROD

    app = FastAPI(
        title="Intergrax Service",
        version="0.1.0",
        docs_url=None if is_prod else "/docs",
        redoc_url=None if is_prod else "/redoc",
        openapi_url=None if is_prod else "/openapi.json",
    )

    # --- Core middleware & handlers ---
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(Exception, global_exception_handler)

    # --- Routers ---
    app.include_router(health_router)
    app.include_router(runs_router)

    # --- Rate limiting ---
    if config.rate_limit_policy is not None:
        app.dependency_overrides[RateLimitPolicy] = (
            lambda: config.rate_limit_policy
        )

    # --- Run store ---
    run_store = (
        config.run_store
        if config.run_store is not None
        else InMemoryRunStore()
    )
    app.dependency_overrides[RunStore] = lambda: run_store

    # --- Auth provider wiring ---
    auth_providers: list[AuthProvider] = []
    
    # Explicit override (advanced / tests / enterprise)
    if config.auth_provider is not None:
        auth_providers.append(config.auth_provider)

    # API key auth (now implemented)
    if config.api_key_config is not None:
        auth_providers.append(ApiKeyAuthProvider(config.api_key_config))

    if not auth_providers:
        auth_provider = NoAuthProvider()
    elif len(auth_providers) == 1:
        auth_provider = auth_providers[0]
    else:
        auth_provider = CompositeAuthProvider(auth_providers)

    app.state.auth_provider = auth_provider


    return app

