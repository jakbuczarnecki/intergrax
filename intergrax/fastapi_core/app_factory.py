# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from fastapi import FastAPI

from intergrax.fastapi_core.auth.api_key import ApiKeyAuthenticator
from intergrax.fastapi_core.config import ApiConfig, ApiEnvironment
from intergrax.fastapi_core.errors.handlers import global_exception_handler
from intergrax.fastapi_core.middleware.request_context import RequestContextMiddleware
from intergrax.fastapi_core.rate_limit.policy import RateLimitPolicy
from intergrax.fastapi_core.routers.health import health_router



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

    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(Exception, global_exception_handler)
    app.include_router(health_router)

    if config.api_key_config is not None:
        authenticator = ApiKeyAuthenticator(config=config.api_key_config)
        app.dependency_overrides[ApiKeyAuthenticator] = (
            lambda: authenticator
        )

    if config.rate_limit_policy is not None:
        app.dependency_overrides[RateLimitPolicy] = (
            lambda: config.rate_limit_policy
        )        

    return app
