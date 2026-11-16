# © Artur Czarnecki. All rights reserved.

"""Test doubles and variant builders for the MCP-SERVER-X reference server (MCP-P7-01).

These are fixtures for qualification tests: a stub identity provider (a real
``IdentityProviderBackend`` protocol implementation, but a *contract stub* and not vendor
integration evidence), a small tool catalog, and a recording catalog invoker.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from intergrax.applications._shared.harness_auth import HarnessAuthState
from intergrax.contracts.execution_bound_catalog_tool_invocation import (
    ExecutionBoundCatalogToolInvokeRequest,
)
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.integrations.contracts.identity_provider import (
    IdentityTenant,
    IdentityUser,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.registry.runtime import ToolRegistry
from pydantic import BaseModel

QUAL_API_KEY = "p7-qual-api-key"
QUAL_API_KEY_TENANT = "p7-tenant-a"
QUAL_TENANT_B = "p7-tenant-b"
CATALOG_TOOL_ID = "p7.catalog.ok"


class StubIdentityProvider:
    """Contract stub: maps opaque bearer tokens to normalized ``IdentityUser`` principals."""

    def __init__(self, users_by_token: Mapping[str, IdentityUser]) -> None:
        self._users = dict(users_by_token)

    def verify_token(self, token: str) -> IdentityUser:
        user = self._users.get(token)
        if user is None:
            raise ValueError("invalid token")
        return user

    def userinfo(self, token: str) -> IdentityUser:
        return self.verify_token(token)

    def list_tenants(self, *, limit: int = 50) -> Sequence[IdentityTenant]:
        return ()


def api_key_auth(
    *,
    api_key: str = QUAL_API_KEY,
    tenant_id: str = QUAL_API_KEY_TENANT,
) -> HarnessAuthState:
    """API-key harness auth bound to one authoritative tenant (``require_api_key=True``)."""
    return HarnessAuthState(
        require_api_key=True,
        resolved_api_key=api_key,
        api_key_principal_tenant_id=tenant_id,
    )


def identity_provider_auth(provider: StubIdentityProvider) -> HarnessAuthState:
    """Bearer-token harness auth through the identity provider (``require_api_key=True``)."""
    return HarnessAuthState(
        identity_provider=provider,
        require_api_key=True,
        tenant_required=True,
    )


def two_tenant_identity_provider() -> StubIdentityProvider:
    return StubIdentityProvider(
        {
            "token-a": IdentityUser(user_id="user-a", tenant_id=QUAL_API_KEY_TENANT),
            "token-b": IdentityUser(user_id="user-b", tenant_id=QUAL_TENANT_B),
        },
    )


class CatalogInput(BaseModel):
    query: str = "x"


class CatalogOutput(BaseModel):
    value: str = "invoked"


def catalog_contract(tool_id: str = CATALOG_TOOL_ID) -> ToolContract:
    return ToolContract(
        tool_id=tool_id,
        name=tool_id,
        description="P7 qualification catalog tool",
        input_schema=CatalogInput,
        output_schema=CatalogOutput,
        error_mapping={},
        side_effects=False,
        risk_level=ToolRiskLevel.LOW,
        category="p7",
    )


def build_catalog_registry(tool_ids: Sequence[str] = (CATALOG_TOOL_ID,)) -> ToolRegistry:
    registry = ToolRegistry()

    async def _handler(_input: CatalogInput, _ctx: object) -> CatalogOutput:
        return CatalogOutput(value="unused")

    for tool_id in tool_ids:
        registry.register(catalog_contract(tool_id), _handler)
    return registry


@dataclass
class RecordingCatalogInvoker:
    """Execution-bound catalog invoker double that records every request."""

    requests: list[ExecutionBoundCatalogToolInvokeRequest] = field(default_factory=list)
    started: threading.Event = field(default_factory=threading.Event)

    def invoke(self, request: ExecutionBoundCatalogToolInvokeRequest) -> ToolExecutionResult[CatalogOutput]:
        self.requests.append(request)
        self.started.set()
        return ToolExecutionResult.ok(CatalogOutput(value="p7-invoked"))
