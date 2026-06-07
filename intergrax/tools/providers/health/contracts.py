# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from pydantic import BaseModel, Field


class HealthCheckIntegrationInput(BaseModel):
    slug: str = Field(..., min_length=1, description="Integration catalog slug to probe.")


class HealthStatusOutput(BaseModel):
    slug: str
    healthy: bool
    detail: str = ""


class HealthCheckIntegrationOutput(BaseModel):
    status: HealthStatusOutput


class HealthCheckProfileInput(BaseModel):
    pass


class HealthCheckProfileOutput(BaseModel):
    statuses: list[HealthStatusOutput] = Field(default_factory=list)
    healthy_count: int = 0
    unhealthy_count: int = 0
