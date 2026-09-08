# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Factory for the security-qualified E2B sandbox host backend."""

from __future__ import annotations

from typing import Any, Callable, Optional

from intergrax.integrations.contracts.sandbox_host import SandboxHostBackend
from intergrax.integrations.providers.sandbox_host.e2b.backend import E2bSandboxHostBackend
from intergrax.integrations.providers.sandbox_host.e2b.client import build_e2b_sandbox_api_client
from intergrax.integrations.providers.sandbox_host.e2b.config import E2bSandboxHostConfig


def create_e2b_sandbox_host_backend(
    *,
    sandbox_host: Optional[SandboxHostBackend] = None,
    client: Optional[Any] = None,
    client_factory: Optional[Callable[[], Any]] = None,
    **config_overrides: object,
) -> E2bSandboxHostBackend:
    """Build the real E2B sandbox host backend (not the generic HTTP shim)."""
    if sandbox_host is not None:
        if not isinstance(sandbox_host, E2bSandboxHostBackend):
            return sandbox_host  # type: ignore[return-value]
        return sandbox_host
    config = E2bSandboxHostConfig.from_env(**config_overrides)
    resolved_client = client if client is not None else (client_factory() if client_factory else None)
    api_client = build_e2b_sandbox_api_client(config, client=resolved_client)
    return E2bSandboxHostBackend(client=api_client, config=config)
