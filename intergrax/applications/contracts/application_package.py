# © Artur Czarnecki. All rights reserved.

"""Tier-3 application packaging contracts (APP-EVOL-7 · §49.7)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.runtime.registry.semver_compat import SemVer


class ApplicationDependencyKind(StrEnum):
    """Direct dependency kind declared on an application package."""

    AGENT = "agent"
    SKILL = "skill"
    TOOL = "tool"
    INTEGRATION = "integration"
    PROFILE_FRAGMENT = "profile_fragment"


class ApplicationDistributionChannel(StrEnum):
    """Package distribution channel."""

    LOCAL = "local"
    GIT = "git"
    REGISTRY = "registry"
    MARKETPLACE = "marketplace"


class ApplicationDependency(BaseModel):
    """One direct dependency entry (§49.7.1)."""

    model_config = ConfigDict(extra="forbid")

    kind: ApplicationDependencyKind
    ref: str = Field(min_length=1)
    version_constraint: str = Field(default="*", min_length=1)
    optional: bool = False


class ApplicationDistribution(BaseModel):
    """Immutable package distribution metadata."""

    model_config = ConfigDict(extra="forbid")

    channel: ApplicationDistributionChannel = ApplicationDistributionChannel.LOCAL
    artifact_uri: str | None = None
    checksum: str = Field(default="", min_length=0)
    signature_ref: str | None = None


class ApplicationPackage(BaseModel):
    """Frozen application package for reproducible deploys (§49.7.1)."""

    model_config = ConfigDict(extra="forbid")

    package_id: str = Field(min_length=1)
    app_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    manifest: ApplicationManifest
    dependencies: list[ApplicationDependency] = Field(default_factory=list)
    distribution: ApplicationDistribution = Field(default_factory=ApplicationDistribution)

    @field_validator("version")
    @classmethod
    def _validate_semver(cls, value: str) -> str:
        SemVer.parse(value)
        return value


class ApplicationPackageClosureError(ValueError):
    """Raised when a package dependency closure check fails."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))
