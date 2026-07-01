# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import ClassVar, FrozenSet, Literal, Mapping, Optional

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.runtime.observability.operator_wiring import (
    ElasticsearchExportOperatorConfig,
    ObservabilityExportOperatorConfig,
    ObservabilityExportOperatorConfigError,
    OtlpExportOperatorConfig,
    parse_observability_export_backend_id,
)



LocalWorkspaceIdentitySource = Literal["body_or_context", "context_only"]


def _parse_api_key_map(raw: Optional[str]) -> Mapping[str, ApiKeyIdentity]:
    if not raw or not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LOCAL_WORKSPACE_BACKEND_API_KEYS_JSON must be a JSON object.")
    out: dict[str, ApiKeyIdentity] = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            raise ValueError(f"Identity for key {key!r} must be an object.")
        tenant = val.get("tenant_id")
        if not tenant or not isinstance(tenant, str):
            raise ValueError(f"tenant_id required for API key {key!r}.")
        user_id = val.get("user_id")
        scopes = val.get("scopes", ("*",))
        if isinstance(scopes, list):
            scopes = tuple(str(s) for s in scopes)
        elif isinstance(scopes, tuple):
            scopes = tuple(str(s) for s in scopes)
        else:
            scopes = ("*",)
        out[str(key)] = ApiKeyIdentity(
            tenant_id=str(tenant),
            user_id=str(user_id) if user_id is not None else None,
            scopes=scopes,
        )
    return out


@dataclass(frozen=True, kw_only=True)
class LocalWorkspaceBackendSettings(IntergraxApplicationSettingsBase):
    """Environment for local_workspace_application (scaffolded product profile)."""

    env_prefix: ClassVar[str] = "LOCAL_WORKSPACE_"
    route_prefix: str = "/v1/local_workspace"
    backend_port: int = 8020
    include_scheduler: bool = False
    include_interaction_routes: bool = False
    default_agent_id: str = "local_search"
    identity_source: LocalWorkspaceIdentitySource = "body_or_context"
    enable_rag: bool = True
    enable_rag_ingest: bool = True
    extra_enabled_tool_ids: tuple[str, ...] = ()
    allowed_read_roots: FrozenSet[str] = field(default_factory=frozenset)
    cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
    openapi_enabled_override: Optional[bool] = None
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
    interaction_execute_default: bool = True
    # Observability export settings (env-driven; disabled by default)
    observability_export_enabled: bool = False
    observability_export_backend: str = "otlp"
    observability_export_content: bool = False
    observability_otlp_endpoint: str = ""
    observability_service_name: str = "intergrax-lkw"
    observability_service_version: str = ""
    observability_environment: str = ""
    observability_otlp_timeout_seconds: float = 30.0
    observability_elasticsearch_url: str = ""
    observability_elasticsearch_index: str = ""
    observability_elasticsearch_timeout_seconds: float = 30.0
    observability_elasticsearch_retry_enabled: bool = True
    observability_elasticsearch_retry_max_attempts: int = 3
    observability_elasticsearch_retry_initial_backoff_seconds: float = 0.25
    observability_elasticsearch_retry_max_backoff_seconds: float = 2.0
    observability_elasticsearch_failed_delivery_file_path: str = ""

    @property
    def enabled_tool_ids(self) -> list[str]:
        """Catalog tool_ids enabled for this host (from env flags)."""
        ids: list[str] = list(self.extra_enabled_tool_ids)
        if self.enable_rag and "rag.retrieve" not in ids:
            ids.append("rag.retrieve")
        if self.enable_rag_ingest and "rag.ingest_document" not in ids:
            ids.append("rag.ingest_document")
        return ids

    # ------------------------------------------------------------------
    def build_observability_export_config(self) -> ObservabilityExportOperatorConfig | None:
        """Build optional ObservabilityExportOperatorConfig from env-driven settings.

        Returns None when export is disabled.  Raises ValueError on
        missing endpoint or unsupported backend.
        """
        if not self.observability_export_enabled:
            return None

        try:
            backend_id = parse_observability_export_backend_id(self.observability_export_backend)
        except ObservabilityExportOperatorConfigError as exc:
            raise ValueError(str(exc)) from exc

        otlp: OtlpExportOperatorConfig | None
        elasticsearch: ElasticsearchExportOperatorConfig | None
        if backend_id == "otlp":
            endpoint = self.observability_otlp_endpoint.strip()
            if not endpoint:
                raise ValueError(
                    "LOCAL_WORKSPACE_OBSERVABILITY_OTLP_ENDPOINT is required when "
                    "observability export is enabled"
                )
            otlp = OtlpExportOperatorConfig(
                endpoint=endpoint,
                service_name=self.observability_service_name or "intergrax-lkw",
                service_version=self.observability_service_version or None,
                environment=self.observability_environment or None,
                timeout_seconds=self.observability_otlp_timeout_seconds,
                headers=None,
            )
            elasticsearch = None
        elif backend_id == "elasticsearch":
            otlp = None
            base_url = self.observability_elasticsearch_url.strip()
            if not base_url:
                raise ValueError(
                    "LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_URL is required when "
                    "observability export is enabled with backend_id=elasticsearch"
                )
            index = self.observability_elasticsearch_index.strip()
            if not index:
                raise ValueError(
                    "LOCAL_WORKSPACE_OBSERVABILITY_ELASTICSEARCH_INDEX is required when "
                    "observability export is enabled with backend_id=elasticsearch"
                )
            failed_delivery_file_path = (
                self.observability_elasticsearch_failed_delivery_file_path.strip() or None
            )
            elasticsearch = ElasticsearchExportOperatorConfig(
                base_url=base_url,
                index=index,
                timeout_seconds=self.observability_elasticsearch_timeout_seconds,
                retry_enabled=self.observability_elasticsearch_retry_enabled,
                retry_max_attempts=self.observability_elasticsearch_retry_max_attempts,
                retry_initial_backoff_seconds=self.observability_elasticsearch_retry_initial_backoff_seconds,
                retry_max_backoff_seconds=self.observability_elasticsearch_retry_max_backoff_seconds,
                failed_delivery_file_path=failed_delivery_file_path,
            )
        else:
            otlp = None
            elasticsearch = None

        # Safety: never export raw content regardless of env value
        return ObservabilityExportOperatorConfig(
            enabled=True,
            export_content=False,
            backend_id=backend_id,
            otlp=otlp,
            elasticsearch=elasticsearch,
        )

    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        env_raw = (
            env.optional_str("BACKEND_ENV")
            or (os.environ.get("INTERGRAX_ENV") or "dev").strip().lower()
        )
        if env_raw == "staging":
            env_raw = "stage"
        environment = ApiEnvironment(env_raw)

        agent_id = env.str("DEFAULT_AGENT_ID", default="local_search") or "local_search"
        enable_rag = env.bool("ENABLE_RAG", default=True)
        enable_rag_ingest = env.bool("ENABLE_RAG_INGEST", default=True)

        id_src_env = env.optional_str("IDENTITY_SOURCE")
        if id_src_env in {"body_or_context", "context_only"}:
            identity_source = id_src_env
        else:
            identity_source = (
                "context_only" if environment == ApiEnvironment.PROD else "body_or_context"
            )

        cors = env.csv_set("BACKEND_CORS_ORIGINS")
        hosts = env.csv_set("BACKEND_ALLOWED_HOSTS")

        openapi_override: Optional[bool] = None
        if env.raw("BACKEND_OPENAPI") is not None:
            openapi_override = env.bool("BACKEND_OPENAPI")

        keys: Mapping[str, ApiKeyIdentity] = {}
        bootstrap_key = env.str("BACKEND_BOOTSTRAP_API_KEY", default="")
        if bootstrap_key:
            tenant = env.str("BACKEND_BOOTSTRAP_TENANT_ID", default="")
            user = env.str("BACKEND_BOOTSTRAP_USER_ID", default="")
            if not tenant or not user:
                raise ValueError(
                    "When LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY is set, "
                    "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_TENANT_ID and "
                    "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_USER_ID are required."
                )
            keys = {
                bootstrap_key: ApiKeyIdentity(
                    tenant_id=tenant,
                    user_id=user,
                    scopes=("*",),
                )
            }
        json_keys = env.str("BACKEND_API_KEYS_JSON", default="")
        if json_keys:
            if keys:
                raise ValueError(
                    "Use either LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY or "
                    "LOCAL_WORKSPACE_BACKEND_API_KEYS_JSON, not both."
                )
            keys = _parse_api_key_map(json_keys)

        if environment == ApiEnvironment.PROD and identity_source != "context_only":
            raise ValueError(
                "LOCAL_WORKSPACE_BACKEND_ENV=prod requires "
                "LOCAL_WORKSPACE_IDENTITY_SOURCE=context_only (or omit to default)."
            )
        if environment == ApiEnvironment.PROD and not env.bool(
            "BACKEND_ALLOW_UNAUTHENTICATED", default=False
        ):
            if not keys:
                raise ValueError(
                    "Production local_workspace backend requires API keys: set "
                    "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY (+ tenant/user) or "
                    "LOCAL_WORKSPACE_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set "
                    "LOCAL_WORKSPACE_BACKEND_ALLOW_UNAUTHENTICATED=true."
                )


        # Observability export
        observability_export_enabled = env.bool(
            "OBSERVABILITY_EXPORT_ENABLED",
            default=cls._field_default("observability_export_enabled"),  # type: ignore[arg-type]
        )
        observability_export_backend = env.str(
            "OBSERVABILITY_EXPORT_BACKEND",
            default=cls._field_default("observability_export_backend"),  # type: ignore[arg-type]
        )
        observability_export_content = env.bool(
            "OBSERVABILITY_EXPORT_CONTENT",
            default=cls._field_default("observability_export_content"),  # type: ignore[arg-type]
        )
        observability_otlp_endpoint = env.str(
            "OBSERVABILITY_OTLP_ENDPOINT",
            default=cls._field_default("observability_otlp_endpoint"),  # type: ignore[arg-type]
        )
        observability_service_name = env.str(
            "OBSERVABILITY_SERVICE_NAME",
            default=cls._field_default("observability_service_name"),  # type: ignore[arg-type]
        )
        observability_service_version = env.str(
            "OBSERVABILITY_SERVICE_VERSION",
            default=cls._field_default("observability_service_version"),  # type: ignore[arg-type]
        )
        observability_environment = env.str(
            "OBSERVABILITY_ENVIRONMENT",
            default=cls._field_default("observability_environment"),  # type: ignore[arg-type]
        )
        observability_otlp_timeout_seconds = env.float(
            "OBSERVABILITY_OTLP_TIMEOUT_SECONDS",
            default=cls._field_default("observability_otlp_timeout_seconds"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_url = env.str(
            "OBSERVABILITY_ELASTICSEARCH_URL",
            default=cls._field_default("observability_elasticsearch_url"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_index = env.str(
            "OBSERVABILITY_ELASTICSEARCH_INDEX",
            default=cls._field_default("observability_elasticsearch_index"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_timeout_seconds = env.float(
            "OBSERVABILITY_ELASTICSEARCH_TIMEOUT_SECONDS",
            default=cls._field_default("observability_elasticsearch_timeout_seconds"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_retry_enabled = env.bool(
            "OBSERVABILITY_ELASTICSEARCH_RETRY_ENABLED",
            default=cls._field_default("observability_elasticsearch_retry_enabled"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_retry_max_attempts = env.int(
            "OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_ATTEMPTS",
            default=cls._field_default("observability_elasticsearch_retry_max_attempts"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_retry_initial_backoff_seconds = env.float(
            "OBSERVABILITY_ELASTICSEARCH_RETRY_INITIAL_BACKOFF_SECONDS",
            default=cls._field_default("observability_elasticsearch_retry_initial_backoff_seconds"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_retry_max_backoff_seconds = env.float(
            "OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_BACKOFF_SECONDS",
            default=cls._field_default("observability_elasticsearch_retry_max_backoff_seconds"),  # type: ignore[arg-type]
        )
        observability_elasticsearch_failed_delivery_file_path = env.str(
            "OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH",
            default=cls._field_default("observability_elasticsearch_failed_delivery_file_path"),  # type: ignore[arg-type]
        )
        read_roots_raw = (os.environ.get("INTERGRAX_ALLOWED_READ_ROOTS") or "").strip()
        allowed_read_roots = frozenset(
            part.strip() for part in read_roots_raw.split(",") if part.strip()
        )

        return {
            "default_agent_id": agent_id,
            "identity_source": identity_source,
            "enable_rag": enable_rag,
            "enable_rag_ingest": enable_rag_ingest,
            "allowed_read_roots": allowed_read_roots,
            "cors_allow_origins": cors,
            "allowed_hosts": hosts,
            "openapi_enabled_override": openapi_override,
            "api_keys_map": keys,
            "interaction_execute_default": env.bool(
                "INTERACTION_EXECUTE_DEFAULT",
                default=cls._field_default("interaction_execute_default"),  # type: ignore[arg-type]
            ),
            "observability_export_enabled": observability_export_enabled,
            "observability_export_backend": observability_export_backend,
            "observability_export_content": observability_export_content,
            "observability_otlp_endpoint": observability_otlp_endpoint,
            "observability_service_name": observability_service_name,
            "observability_service_version": observability_service_version,
            "observability_environment": observability_environment,
            "observability_otlp_timeout_seconds": observability_otlp_timeout_seconds,
            "observability_elasticsearch_url": observability_elasticsearch_url,
            "observability_elasticsearch_index": observability_elasticsearch_index,
            "observability_elasticsearch_timeout_seconds": observability_elasticsearch_timeout_seconds,
            "observability_elasticsearch_retry_enabled": observability_elasticsearch_retry_enabled,
            "observability_elasticsearch_retry_max_attempts": observability_elasticsearch_retry_max_attempts,
            "observability_elasticsearch_retry_initial_backoff_seconds": observability_elasticsearch_retry_initial_backoff_seconds,
            "observability_elasticsearch_retry_max_backoff_seconds": observability_elasticsearch_retry_max_backoff_seconds,
            "observability_elasticsearch_failed_delivery_file_path": observability_elasticsearch_failed_delivery_file_path,
        }
