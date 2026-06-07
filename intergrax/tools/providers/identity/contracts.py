# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class IdentityVerifyTokenInput(BaseModel):
    token: str = Field(..., min_length=1)


class IdentityUserOutput(BaseModel):
    user_id: str
    email: str = ""
    name: str = ""
    tenant_id: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class IdentityVerifyTokenOutput(BaseModel):
    valid: bool = True
    user: IdentityUserOutput


class IdentityGetUserInput(BaseModel):
    token: str = Field(..., min_length=1)


class IdentityGetUserOutput(BaseModel):
    user: IdentityUserOutput


class IdentityListTenantsInput(BaseModel):
    limit: int = Field(default=50, ge=1, le=200)


class IdentityTenantOutput(BaseModel):
    tenant_id: str
    name: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class IdentityListTenantsOutput(BaseModel):
    tenants: list[IdentityTenantOutput] = Field(default_factory=list)
    total: int = 0
