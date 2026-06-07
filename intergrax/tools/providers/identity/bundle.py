# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.identity.contracts import (
    IdentityGetUserInput,
    IdentityGetUserOutput,
    IdentityListTenantsInput,
    IdentityListTenantsOutput,
    IdentityVerifyTokenInput,
    IdentityVerifyTokenOutput,
)
from intergrax.tools.providers.identity.handlers import (
    IdentityGetUserHandler,
    IdentityListTenantsHandler,
    IdentityVerifyTokenHandler,
)
from intergrax.tools.providers.identity.service import (
    IDENTITY_GET_USER_TOOL_ID,
    IDENTITY_LIST_TENANTS_TOOL_ID,
    IDENTITY_VERIFY_TOKEN_TOOL_ID,
)
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

IDENTITY_BUNDLE_ID = "identity"
IDENTITY_TOOL_IDS: tuple[str, ...] = (
    IDENTITY_VERIFY_TOKEN_TOOL_ID,
    IDENTITY_GET_USER_TOOL_ID,
    IDENTITY_LIST_TENANTS_TOOL_ID,
)


def register_identity_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=IDENTITY_VERIFY_TOKEN_TOOL_ID,
            name=IDENTITY_VERIFY_TOKEN_TOOL_ID,
            description="Verify a bearer token via the configured identity provider and return user profile.",
            description_short="Verify OIDC token.",
            input_schema=IdentityVerifyTokenInput,
            output_schema=IdentityVerifyTokenOutput,
            error_mapping={},
            side_effects=False,
            category="identity",
            risk_level=ToolRiskLevel.LOW,
            tags=("identity", "auth", "sso"),
        ),
        IdentityVerifyTokenHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=IDENTITY_GET_USER_TOOL_ID,
            name=IDENTITY_GET_USER_TOOL_ID,
            description="Fetch user profile for a valid access token from the identity provider.",
            description_short="Get user profile.",
            input_schema=IdentityGetUserInput,
            output_schema=IdentityGetUserOutput,
            error_mapping={},
            side_effects=False,
            category="identity",
            risk_level=ToolRiskLevel.LOW,
            tags=("identity", "auth", "sso"),
        ),
        IdentityGetUserHandler(ctx),
    )
    registry.register(
        ToolContract(
            tool_id=IDENTITY_LIST_TENANTS_TOOL_ID,
            name=IDENTITY_LIST_TENANTS_TOOL_ID,
            description="List organization tenants from the configured identity provider directory.",
            description_short="List tenants.",
            input_schema=IdentityListTenantsInput,
            output_schema=IdentityListTenantsOutput,
            error_mapping={},
            side_effects=False,
            category="identity",
            risk_level=ToolRiskLevel.MEDIUM,
            tags=("identity", "directory", "sso"),
        ),
        IdentityListTenantsHandler(ctx),
    )
