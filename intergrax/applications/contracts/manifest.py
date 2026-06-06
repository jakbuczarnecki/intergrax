# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Declarative Tier-3 application composition contract (Phase N.1+).

Prefer :meth:`AgentBinding.mount` with ``type[Agent]`` and typed ``factory`` callables.
String ``import_path`` / ``factory_path`` remain for scaffold-generated manifests only.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.agent_ref import qualname_for_agent, qualname_for_callable
from intergrax.applications.contracts.application_host import (
    ApplicationFeatures,
    ApplicationProfile,
)
from intergrax.integrations.registry.profile import IntegrationProfile

_APP_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ENV_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]*_$")
_IMPORT_PATH_RE = re.compile(
    r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]+)+\.[A-Z][a-zA-Z0-9_]*$"
)
_FACTORY_PATH_RE = re.compile(
    r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]+)+\.[a-z][a-z0-9_]*$"
)


class AgentBinding(BaseModel):
    """
    One agent entry in the application roster.

    **Preferred (strongly typed):** ::

        AgentBinding.mount(EchoAgent)
        AgentBinding.mount(LegalAgent, factory=build_legal_agent)

    **Serialized (scaffold / YAML only):** ``deserialize(import_path=..., factory_path=...)``

    Instance creation order in :func:`~intergrax.applications._shared.wiring.build_agent_from_binding`:

    1. ``factory`` callable (typed)
    2. ``builders[agent_type]`` or ``builders[builder_key]``
    3. ``factory_path`` (legacy string import)
    4. zero-argument ``agent_type()`` constructor
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    agent_type: SkipJsonSchema[type[Agent] | None] = Field(
        default=None,
        description="Tier-2 agent class (preferred — checked at authoring time)",
        exclude=True,
    )
    import_path: str | None = Field(
        default=None,
        description="Serialized class path; auto-filled from agent_type when mounting",
    )
    factory: SkipJsonSchema[Callable[..., Any] | None] = Field(
        default=None,
        description="Typed Tier-3 factory callable (preferred over factory_path)",
        exclude=True,
    )
    factory_path: str | None = Field(
        default=None,
        description="Serialized factory path — scaffold/YAML only",
    )
    builder_key: str | None = Field(
        default=None,
        description="Fallback key into builders map (prefer type-keyed builders)",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="Lightweight binding options consumed by factories (not secrets)",
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
    requires_uaep: bool = Field(
        default=False,
        description="When true, registration fails unless the agent implements UAEPAgent",
    )
    default: bool = Field(
        default=False,
        description="When true, used as default agent for product-style routes",
    )

    @classmethod
    def mount(
        cls,
        agent_type: type[Agent],
        *,
        factory: Callable[..., Any] | None = None,
        builder_key: str | None = None,
        config: dict[str, Any] | None = None,
        contract_id: str | None = None,
        capabilities: list[str] | None = None,
        enabled: bool = True,
        default: bool = False,
        requires_uaep: bool = False,
    ) -> AgentBinding:
        """
        Strongly-typed roster entry — pass the agent **class** and optional **factory**.

        Example::

            from legal.legal_agent import LegalAgent
            from legal_application.host.wiring import build_legal_agent

            AgentBinding.mount(LegalAgent, factory=build_legal_agent, default=True)
        """
        if factory is not None and builder_key is not None:
            raise ValueError("AgentBinding.mount: pass factory or builder_key, not both")
        return cls(
            agent_type=agent_type,
            import_path=qualname_for_agent(agent_type),
            factory=factory,
            builder_key=builder_key,
            config=config or {},
            contract_id=contract_id,
            capabilities=capabilities or [],
            enabled=enabled,
            default=default,
            requires_uaep=requires_uaep,
        )

    @classmethod
    def deserialize(
        cls,
        *,
        import_path: str,
        factory_path: str | None = None,
        builder_key: str | None = None,
        config: dict[str, Any] | None = None,
        contract_id: str | None = None,
        capabilities: list[str] | None = None,
        enabled: bool = True,
        default: bool = False,
    ) -> AgentBinding:
        """Load roster entry from strings (scaffold / generated manifests only)."""
        return cls(
            import_path=import_path,
            factory_path=factory_path,
            builder_key=builder_key,
            config=config or {},
            contract_id=contract_id,
            capabilities=capabilities or [],
            enabled=enabled,
            default=default,
        )

    def resolved_agent_type(self) -> type[Agent]:
        from intergrax.applications.contracts.agent_ref import resolve_agent_type

        return resolve_agent_type(agent_type=self.agent_type, import_path=self.import_path)

    def display_name(self) -> str:
        if self.agent_type is not None:
            return self.agent_type.__name__
        return self.import_path or "<unknown>"

    @field_validator("import_path")
    @classmethod
    def _validate_import_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
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

    @field_validator("factory_path")
    @classmethod
    def _validate_factory_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = value.strip()
        if not path:
            return None
        if not _FACTORY_PATH_RE.match(path):
            raise ValueError(
                "factory_path must look like 'package.module.build_agent' "
                "(lowercase module path and function name)"
            )
        return path

    @field_validator("builder_key")
    @classmethod
    def _strip_builder_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None

    @field_validator("capabilities")
    @classmethod
    def _normalize_capabilities(cls, value: list[str]) -> list[str]:
        return [c.strip() for c in value if c and c.strip()]

    @model_validator(mode="after")
    def _normalize_typed_fields(self) -> AgentBinding:
        if self.agent_type is not None and self.import_path is None:
            object.__setattr__(self, "import_path", qualname_for_agent(self.agent_type))
        if self.factory is not None and self.factory_path is None:
            object.__setattr__(self, "factory_path", qualname_for_callable(self.factory))

        if self.agent_type is None and self.import_path is None:
            raise ValueError("AgentBinding requires agent_type (mount) or import_path (deserialize)")

        if self.builder_key and self.factory is not None:
            raise ValueError("AgentBinding: factory and builder_key are mutually exclusive")

        return self


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
    environment: "ApplicationEnvironmentProfile | None" = Field(
        default=None,
        description="Optional IDEAL §17 environment umbrella (Phase H-APP.1.2)",
    )

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

    def resolved_environment(self) -> "ApplicationEnvironmentProfile":
        """Return manifest environment or profile-appropriate defaults."""
        from intergrax.applications.contracts.environment_profile import (
            ApplicationEnvironmentProfile,
        )

        if self.environment is not None:
            return self.environment
        if self.profile is ApplicationProfile.PRODUCT:
            return ApplicationEnvironmentProfile.product_defaults()
        return ApplicationEnvironmentProfile.lab_defaults()

    @classmethod
    def environment_defaults(cls, profile: ApplicationProfile) -> "ApplicationEnvironmentProfile":
        """Factory for ``lab`` / ``product`` environment presets (H-APP.1.2)."""
        from intergrax.applications.contracts.environment_profile import (
            ApplicationEnvironmentProfile,
        )

        if profile is ApplicationProfile.PRODUCT:
            return ApplicationEnvironmentProfile.product_defaults()
        return ApplicationEnvironmentProfile.lab_defaults()

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


def _rebuild_application_manifest_model() -> None:
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    ApplicationManifest.model_rebuild(
        _types_namespace={"ApplicationEnvironmentProfile": ApplicationEnvironmentProfile},
    )


_rebuild_application_manifest_model()
