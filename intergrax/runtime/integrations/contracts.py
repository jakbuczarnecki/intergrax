# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Generic platform integration contract (INTEGRATIONS-1A)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

PLATFORM_INTEGRATION_CONTRACT_SCHEMA = "platform_integration_contract.v1"

_SECRET_CONFIG_FIELD_NAMES: frozenset[str] = frozenset(
    {
        "secret",
        "secrets",
        "password",
        "token",
        "api_key",
        "api_secret",
        "access_token",
        "refresh_token",
        "private_key",
        "credentials",
    }
)


class PlatformIntegrationKind(StrEnum):
    """
    Well-known integration categories — one provider may appear in many kinds.

    Provider folder taxonomy (`layout.py` SLUG_CATEGORY) uses snake_case category
    strings (for example ``observability_backend``, ``search_provider``). Legacy
  shorthand values (``search``, ``storage``, ``notification``) remain for backward
    compatibility. Observability vendor integrations use ``observability_vendor`` as
    ``integration_kind`` even when the provider folder is ``observability_backend``.
    """

    # Legacy shorthand values (backward compatible — do not remove)
    OBSERVABILITY_VENDOR = "observability_vendor"
    LLM_PROVIDER = "llm_provider"
    VECTOR_STORE = "vector_store"
    SEARCH = "search"
    STORAGE = "storage"
    NOTIFICATION = "notification"
    TOOL_PROVIDER = "tool_provider"

    # Provider category taxonomy (layout.py SLUG_CATEGORY)
    RELATIONAL_STORE = "relational_store"
    DOCUMENT_STORE = "document_store"
    KEY_VALUE_CACHE = "key_value_cache"
    MESSAGE_BUS = "message_bus"
    OBJECT_STORAGE = "object_storage"
    SEARCH_PROVIDER = "search_provider"
    NOTIFICATION_CHANNEL = "notification_channel"
    INTERACTION_SURFACE = "interaction_surface"
    COLLABORATION_SUITE = "collaboration_suite"
    ISSUE_TRACKER = "issue_tracker"
    WIKI_KNOWLEDGE = "wiki_knowledge"
    OBSERVABILITY_BACKEND = "observability_backend"
    BROWSER_AUTOMATION = "browser_automation"
    CLOUD_PLATFORM = "cloud_platform"
    SECRETS_STORE = "secrets_store"
    GRAPH_STORE = "graph_store"
    DOCUMENT_PARSER = "document_parser"
    RERANK_PROVIDER = "rerank_provider"
    FEATURE_FLAG = "feature_flag"
    CI_CD = "ci_cd"
    SECURITY_SCANNER = "security_scanner"
    SANDBOX_HOST = "sandbox_host"
    IDENTITY_PROVIDER = "identity_provider"
    SPEECH_PROVIDER = "speech_provider"
    WORKFLOW_ORCHESTRATOR = "workflow_orchestrator"
    BILLING_METER = "billing_meter"
    CRM = "crm"
    VISION_SERVING = "vision_serving"
    ML_INFERENCE_HOST = "ml_inference_host"
    LLM_GUARDRAIL = "llm_guardrail"


class PlatformIntegrationCapability(StrEnum):
    """Minimal capability tokens shared across integration categories."""

    CONNECT = "connect"
    HEALTH_CHECK = "health_check"
    EXPORT = "export"
    READ = "read"
    WRITE = "write"


class PlatformIntegrationStatus(StrEnum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class PlatformIntegrationSecurityPosture(BaseModel):
    """Default-safe exposure rules for integration diagnostics and config views."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    expose_secrets: bool = False
    expose_raw_payloads: bool = False
    sanitize_diagnostics: bool = True

    def public_view(self) -> Mapping[str, bool]:
        """Return a safe, non-secret diagnostic view of the posture."""
        return {
            "expose_secrets": self.expose_secrets,
            "expose_raw_payloads": self.expose_raw_payloads,
            "sanitize_diagnostics": self.sanitize_diagnostics,
        }


class PlatformIntegrationConfig(BaseModel):
    """Typed base config for explicit opt-in platform integrations."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    timeout_seconds: float | None = None

    def public_view(self) -> Mapping[str, Any]:
        """Return config fields safe for logs, health checks, and operator UIs."""
        data = self.model_dump(exclude_none=True)
        for key in list(data):
            if key in _SECRET_CONFIG_FIELD_NAMES or key.endswith("_secret") or key.endswith("_token"):
                data.pop(key, None)
        return data


class PlatformIntegrationHealth(BaseModel):
    """Lightweight health/check snapshot — not a full lifecycle manager."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: PlatformIntegrationStatus = PlatformIntegrationStatus.UNKNOWN
    message: str = ""
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


def derive_platform_integration_id(provider_id: str, integration_kind: str) -> str:
    """Stable integration identity — category-specific, separate from provider identity."""
    return f"{provider_id}:{integration_kind}"


class PlatformIntegrationContract(BaseModel):
    """
    Generic platform integration contract.

    Category-specific contracts (for example observability vendor integrations)
    should derive from this base type. Concrete providers must not multi-inherit
    unrelated category contracts — use one integration class per category.
    """

    model_config = ConfigDict(extra="forbid")

    schema_id: Literal["platform_integration_contract.v1"] = PLATFORM_INTEGRATION_CONTRACT_SCHEMA
    integration_id: str
    provider_id: str
    integration_kind: str
    display_name: str | None = None
    version: str | None = None
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(default_factory=tuple)
    config: PlatformIntegrationConfig = Field(default_factory=PlatformIntegrationConfig)
    security_posture: PlatformIntegrationSecurityPosture = Field(
        default_factory=PlatformIntegrationSecurityPosture
    )
    expects_failure_isolation: bool = True

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        integration_kind: str | PlatformIntegrationKind,
        capabilities: tuple[PlatformIntegrationCapability, ...] = (),
        display_name: str | None = None,
        version: str | None = None,
        config: PlatformIntegrationConfig | None = None,
    ) -> PlatformIntegrationContract:
        kind_value = (
            integration_kind.value
            if isinstance(integration_kind, PlatformIntegrationKind)
            else integration_kind
        )
        return cls(
            integration_id=derive_platform_integration_id(provider_id, kind_value),
            provider_id=provider_id,
            integration_kind=kind_value,
            display_name=display_name,
            version=version,
            capabilities=capabilities,
            config=config or PlatformIntegrationConfig(),
        )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def check_health(self) -> PlatformIntegrationHealth:
        if not self.config.enabled:
            return PlatformIntegrationHealth(
                status=PlatformIntegrationStatus.DISABLED,
                message="integration is disabled",
            )
        if PlatformIntegrationCapability.HEALTH_CHECK in self.capabilities:
            return PlatformIntegrationHealth(status=PlatformIntegrationStatus.UNKNOWN)
        return PlatformIntegrationHealth(
            status=PlatformIntegrationStatus.UNKNOWN,
            message="health check capability not declared",
        )

    def public_view(self) -> Mapping[str, Any]:
        """Safe operator-facing view — no secrets or raw payloads by default."""
        view: dict[str, Any] = {
            "schema_id": self.schema_id,
            "integration_id": self.integration_id,
            "provider_id": self.provider_id,
            "integration_kind": self.integration_kind,
            "enabled": self.enabled,
            "capabilities": [capability.value for capability in self.capabilities],
            "expects_failure_isolation": self.expects_failure_isolation,
            "security_posture": dict(self.security_posture.public_view()),
            "config": dict(self.config.public_view()),
        }
        if self.display_name is not None:
            view["display_name"] = self.display_name
        if self.version is not None:
            view["version"] = self.version
        return view
