# © Artur Czarnecki. All rights reserved.

"""Platform ops registries for Tier-3 applications and environments (APP-OPS-4 · §50.4)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.application_package import ApplicationPackage
from intergrax.applications.contracts.environment_health_score import (
    ApplicationHealthScore,
    EnvironmentHealthScore,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.operational_ownership import ApplicationOperationalOwnership
from intergrax.runtime.registry.semver_compat import SemVer


class ApplicationRegistrySource(StrEnum):
    """How an application entry entered the registry."""

    GIT = "git"
    MANUAL = "manual"
    MARKETPLACE = "marketplace"


class EnvironmentDeploymentChannel(StrEnum):
    """Deployment channel for a registered environment."""

    LOCAL = "local"
    DOCKER = "docker"
    K8S = "k8s"
    SERVERLESS = "serverless"


class EnvironmentDeployment(BaseModel):
    """Deployment metadata for one environment instance (§50.4.2)."""

    model_config = ConfigDict(extra="forbid")

    channel: EnvironmentDeploymentChannel = EnvironmentDeploymentChannel.DOCKER
    region: str | None = None
    image_tag: str | None = None
    endpoint: str | None = None
    deployed_at: datetime
    deployed_by: str = Field(min_length=1)


class ApplicationRegistryEntry(BaseModel):
    """One application inventory row (§50.4.1)."""

    model_config = ConfigDict(extra="forbid")

    app_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    current_version: str = Field(min_length=1)
    package_ref: ApplicationPackage | None = None
    ownership: ApplicationOperationalOwnership
    health: ApplicationHealthScore | None = None
    registered_at: datetime
    source: ApplicationRegistrySource = ApplicationRegistrySource.GIT

    @field_validator("current_version")
    @classmethod
    def _validate_semver(cls, value: str) -> str:
        SemVer.parse(value)
        return value


class ApplicationRegistry(BaseModel):
    """Inventory of Tier-3 application packages."""

    model_config = ConfigDict(extra="forbid")

    entries: list[ApplicationRegistryEntry] = Field(default_factory=list)

    def get(self, app_id: str) -> ApplicationRegistryEntry | None:
        for entry in self.entries:
            if entry.app_id == app_id:
                return entry
        return None


class EnvironmentRegistryEntry(BaseModel):
    """One deployed environment instance (§50.4.2)."""

    model_config = ConfigDict(extra="forbid")

    environment_id: str = Field(min_length=1)
    app_id: str = Field(min_length=1)
    app_version: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    execution_mode: ExecutionMode
    deployment: EnvironmentDeployment
    snapshot_id: str | None = None
    health: EnvironmentHealthScore | None = None

    @field_validator("app_version")
    @classmethod
    def _validate_app_version(cls, value: str) -> str:
        SemVer.parse(value)
        return value


class EnvironmentRegistry(BaseModel):
    """Inventory of deployed Tier-3 environments."""

    model_config = ConfigDict(extra="forbid")

    entries: list[EnvironmentRegistryEntry] = Field(default_factory=list)

    def get(self, environment_id: str) -> EnvironmentRegistryEntry | None:
        for entry in self.entries:
            if entry.environment_id == environment_id:
                return entry
        return None

    def list_for_app(self, app_id: str) -> list[EnvironmentRegistryEntry]:
        return [entry for entry in self.entries if entry.app_id == app_id]
