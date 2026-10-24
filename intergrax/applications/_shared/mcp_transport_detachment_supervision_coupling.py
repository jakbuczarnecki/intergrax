# © Artur Czarnecki. All rights reserved.

"""Resolve harness transport-detachment supervision for MCP host coupling."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from intergrax.applications._shared.fastapi_lifespan import LifespanFn
from intergrax.contracts.root_execution_transport_detachment_supervision import (
    RootExecutionTransportDetachmentSupervisionPort,
)
from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
    _DEFAULT_SHUTDOWN_DRAIN_S,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from starlette.requests import Request

    from intergrax.applications._shared.harness_host_runtime import HarnessHostRuntime
    from intergrax.runtime.execution.root_execution_transport_detachment_supervision import (
        HostRootExecutionTransportDetachmentSupervisor,
    )


def resolve_mcp_coupling_transport_detachment_supervision(
    *,
    runtime: HarnessHostRuntime | None = None,
    supervision: HostRootExecutionTransportDetachmentSupervisor | None = None,
) -> RootExecutionTransportDetachmentSupervisionPort | None:
    if supervision is not None:
        return supervision
    if runtime is not None:
        return runtime.transport_detachment_supervision
    return None


def resolve_transport_detachment_supervision_from_request(
    request: Request | None = None,
) -> RootExecutionTransportDetachmentSupervisionPort | None:
    from intergrax.applications._shared.mcp_harness_principal import (
        _starlette_request_from_fastmcp,
    )

    if request is None:
        request = _starlette_request_from_fastmcp()
    if request is None:
        return None
    return getattr(
        request.app.state,
        "root_execution_transport_detachment_supervision",
        None,
    )


def build_transport_detachment_supervision_shutdown_lifespan(
    supervision: RootExecutionTransportDetachmentSupervisionPort,
    *,
    timeout_s: float = _DEFAULT_SHUTDOWN_DRAIN_S,
) -> LifespanFn:
    @asynccontextmanager
    async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await supervision.shutdown_supervised_work(timeout_s=timeout_s)

    return _lifespan


__all__ = [
    "build_transport_detachment_supervision_shutdown_lifespan",
    "resolve_mcp_coupling_transport_detachment_supervision",
    "resolve_transport_detachment_supervision_from_request",
]
