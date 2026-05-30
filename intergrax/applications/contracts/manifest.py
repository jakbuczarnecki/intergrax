# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Declarative Tier-3 application composition contract (Phase N.1).

Canon: intergrax_runtime_architecture.md §7.4.10

Describes which agents, integrations, and host features are active in an
application environment — not domain logic (that stays in Tier-2 ``agents/``).
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.registry.profile import IntegrationProfile

_APP_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENV_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]*_$")
_IMPORT_PATH_RE = re.compile(
    r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]+)+\.[A-Z][a-zA-Z0-9_]*$"
)


class ApplicationProfile(str, Enum):
    """Scaffold / factory profile for Tier-3 hosts."""

    LAB = "lab"
    PRODUCT = "product"


class AgentBinding(BaseModel):
    """
    One agent entry in the application roster.

    ``import_path`` is the fully-qualified class path used at wiring time, e.g.
    ``echo.echo_agent.EchoAgent`` (resolved via ``agents/`` on ``pythonpath``).
    """

    model_config = ConfigDict(extra="forbid")

    import_path: str = Field(
        ...,
        description="Fully-qualified agent class: package.module.ClassName",
    )
    contract_id: str | None = Field(
        default=None,
        description="Optional override for AgentContract.id at register time",
    )
    capabilities: list[str] = Field(
        default_factory=list,
        description="Documented capability ids (routing hints; not enforced here)",
    )
    enabled: bool = True
    default: bool = Field(
        default=False,
        description="When true, used as default agent for product-style routes",
    )

    @field_validator("import_path")
    @classmethod
    def _validate_import_path(cls, value: str) -> str:
        path = value.strip()
        if not _IMPORT_PATH_RE.match(path):
            raise ValueError(
                "import_path must look like 'package.module.ClassName' "
                "(lowercase module path, PascalCase class)"
            )
        return path

    @field_validator("contract_id")
    @classmethod
    def _strip_contract_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("capabilities")
    @classmethod
    def _normalize_capabilities(cls, value: list[str]) -> list[str]:
        return [c.strip() for c in value if c and c.strip()]


class ApplicationFeatures(BaseModel):
    """Product/lab behaviors toggled by the Tier-3 host factory."""

    model_config = ConfigDict(extra="forbid")

    debug_surface: bool = Field(
        default=True,
        description="Expose /debug/* inspection routes (lab profile default)",
    )
    interaction_routes: bool = Field(
        default=True,
        description="Mount inbound interaction intake router",
    )
    long_running_scheduler: bool = Field(
        default=True,
        description="Start long-running scheduler on app startup",
    )
    openapi: bool | None = Field(
        default=None,
        description="Override OpenAPI exposure; None = profile default",
    )
    task_sandbox_default: bool = Field(
        default=False,
        description="Default Task metadata sandbox flag (Tier-1 isolation, not this host)",
    )

    @classmethod
    def lab_defaults(cls) -> ApplicationFeatures:
        return cls(
            debug_surface=True,
            interaction_routes=True,
            long_running_scheduler=True,
            openapi=None,
            task_sandbox_default=False,
        )

    @classmethod
    def product_defaults(cls) -> ApplicationFeatures:
        return cls(
            debug_surface=False,
            interaction_routes=False,
            long_running_scheduler=False,
            openapi=False,
            task_sandbox_default=False,
        )


class ApplicationManifest(BaseModel):
    """
    Tier-3 composition contract for a deployable application environment.

    Wired by ``applications/<app>/host/wiring.py`` and/or scaffold output.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: str = Field(..., description="Stable slug, e.g. my_lab")
    name: str = Field(..., description="Human-readable application title")
    description: str = ""
    version: str = "0.1.0"
    profile: ApplicationProfile = ApplicationProfile.LAB
    route_prefix: str = Field(..., description="HTTP API prefix, e.g. /v1/my_lab")
    env_prefix: str = Field(
        ...,
        description="Environment variable prefix for this app, e.g. MY_LAB_",
    )
    default_host: str = "127.0.0.1"
    default_port: int = Field(default=8090, ge=1, le=65535)
    default_capability: str | None = Field(
        default=None,
        description="Optional default capability for run endpoints",
    )
    agents: list[AgentBinding] = Field(default_factory=list)
    integration_profile: IntegrationProfile = Field(
        default_factory=IntegrationProfile.lab,
    )
    features: ApplicationFeatures = Field(default_factory=ApplicationFeatures.lab_defaults)

    @field_validator("app_id")
    @classmethod
    def _validate_app_id(cls, value: str) -> str:
        slug = value.strip().lower()
        if not _APP_ID_RE.match(slug):
            raise ValueError("app_id must be lowercase slug: [a-z][a-z0-9_]*")
        return slug

    @field_validator("route_prefix")
    @classmethod
    def _validate_route_prefix(cls, value: str) -> str:
        prefix = value.strip()
        if not prefix.startswith("/"):
            raise ValueError("route_prefix must start with '/'")
        return prefix.rstrip("/") or "/"

    @field_validator("env_prefix")
    @classmethod
    def _validate_env_prefix(cls, value: str) -> str:
        prefix = value.strip().upper()
        if not prefix.endswith("_"):
            prefix = f"{prefix}_"
        if not _ENV_PREFIX_RE.match(prefix):
            raise ValueError("env_prefix must be uppercase letters/digits ending with '_'")
        return prefix

    @field_validator("default_capability")
    @classmethod
    def _strip_default_capability(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @model_validator(mode="after")
    def _single_default_agent(self) -> ApplicationManifest:
        defaults = [b for b in self.agents if b.default and b.enabled]
        if len(defaults) > 1:
            raise ValueError("at most one AgentBinding may have default=True")
        return self

    def enabled_agents(self) -> list[AgentBinding]:
        """Bindings with ``enabled=True``."""
        return [b for b in self.agents if b.enabled]

    def default_agent(self) -> AgentBinding | None:
        """First enabled binding marked ``default=True``, if any."""
        for binding in self.agents:
            if binding.enabled and binding.default:
                return binding
        return None

    def require_enabled_agents(self) -> None:
        """Raise when no agents are enabled (runtime wiring guard)."""
        if not self.enabled_agents():
            raise ValueError(f"ApplicationManifest {self.app_id!r} has no enabled agents")

    @classmethod
    def lab(
        cls,
        *,
        app_id: str,
        name: str,
        agents: list[AgentBinding],
        route_prefix: str = "/v1/lab",
        env_prefix: str = "LAB_",
        description: str = "",
        default_port: int = 8090,
        integration_profile: IntegrationProfile | None = None,
        **kwargs: Any,
    ) -> ApplicationManifest:
        """Convenience constructor matching ``lab_application`` conventions."""
        return cls(
            app_id=app_id,
            name=name,
            description=description,
            profile=ApplicationProfile.LAB,
            route_prefix=route_prefix,
            env_prefix=env_prefix,
            default_port=default_port,
            agents=agents,
            integration_profile=integration_profile or IntegrationProfile.lab(),
            features=ApplicationFeatures.lab_defaults(),
            **kwargs,
        )

    @classmethod
    def product(
        cls,
        *,
        app_id: str,
        name: str,
        agents: list[AgentBinding],
        route_prefix: str,
        env_prefix: str,
        description: str = "",
        default_port: int = 8000,
        integration_profile: IntegrationProfile | None = None,
        **kwargs: Any,
    ) -> ApplicationManifest:
        """Convenience constructor for product-style Tier-3 hosts."""
        return cls(
            app_id=app_id,
            name=name,
            description=description,
            profile=ApplicationProfile.PRODUCT,
            route_prefix=route_prefix,
            env_prefix=env_prefix,
            default_port=default_port,
            agents=agents,
            integration_profile=integration_profile or IntegrationProfile(),
            features=ApplicationFeatures.product_defaults(),
            **kwargs,
        )
