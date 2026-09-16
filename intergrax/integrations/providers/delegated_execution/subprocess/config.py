# © Artur Czarnecki. All rights reserved.

"""Typed configuration for subprocess delegated execution provider."""

from __future__ import annotations

from pydantic import ConfigDict, Field

from intergrax.runtime.integrations.contracts import PlatformIntegrationConfig


class SubprocessDelegatedExecutionProviderConfig(PlatformIntegrationConfig):
    """Immutable integration config for subprocess delegated execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_timeout_seconds: float = Field(default=30.0, gt=0.0)
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0)
    max_concurrent_requests: int = Field(default=8, ge=1, le=64)
    connection_auth_token: str | None = Field(default=None, min_length=1)
