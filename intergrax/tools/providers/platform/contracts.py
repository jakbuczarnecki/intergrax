# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class PlatformGetSecretInput(BaseModel):
    path: str = Field(..., min_length=1)
    version: str | None = None


class PlatformGetSecretOutput(BaseModel):
    path: str
    value: str


class PlatformPutSecretInput(BaseModel):
    path: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class PlatformPutSecretOutput(BaseModel):
    path: str
    stored: bool = True


class PlatformEvaluateFeatureFlagInput(BaseModel):
    flag_key: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    user_id: str = ""


class PlatformFeatureFlagOutput(BaseModel):
    key: str
    enabled: bool
    variant: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class PlatformGetWorkflowRunInput(BaseModel):
    run_id: str = Field(..., min_length=1)


class PlatformWorkflowRunOutput(BaseModel):
    id: str
    name: str = ""
    status: str = ""
    conclusion: str = ""
    url: str = ""


class PlatformListCheckSuitesInput(BaseModel):
    ref: str = Field(..., min_length=1, description="Git ref (branch, tag, or commit sha).")
    limit: int = Field(default=20, ge=1, le=100)


class PlatformCheckSuiteOutput(BaseModel):
    id: str
    name: str = ""
    status: str = ""
    conclusion: str = ""
    url: str = ""


class PlatformListCheckSuitesOutput(BaseModel):
    ref: str
    suites: list[PlatformCheckSuiteOutput] = Field(default_factory=list)
    total: int = 0
