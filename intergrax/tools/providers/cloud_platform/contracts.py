# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class CloudPlatformHealthInput(BaseModel):
    tenant_id: str = Field(default="default")


class CloudPlatformHealthOutput(BaseModel):
    used: bool = False
    slug: str = ""
    healthy: bool = False
    default_region: str = ""
    detail: str = ""


class CloudPlatformResolveInput(BaseModel):
    category: str = Field(..., min_length=1, description="Integration category slug, e.g. object_storage.")
    tenant_id: str = Field(default="default")


class CloudPlatformResolveOutput(BaseModel):
    used: bool = False
    category: str = ""
    resolved_slug: str = ""
    reason: str = ""
