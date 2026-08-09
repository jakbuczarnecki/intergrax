# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, FrozenSet, Literal, Mapping, Optional

from intergrax.applications.contracts.settings import (
    EnvReader,
    IntergraxApplicationSettingsBase,
)
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment
from intergrax.runtime.observability.operator_wiring import (
    ElasticsearchExportOperatorConfig,
    ObservabilityExportOperatorConfig,
    ObservabilityExportOperatorConfigError,
    OtlpExportOperatorConfig,
    SentryExportOperatorConfig,
    parse_observability_export_backend_id,
)


LocalWorkspaceIdentitySource = Literal["body_or_context", "context_only"]

_DEFAULT_DATA_HOME = "build/local_workspace"


def _resolve_data_home(env: EnvReader) -> str:
    primary = env.optional_str("DATA_HOME")
    if primary is not None:
        return primary
    legacy = (os.environ.get("LKW_DATA_HOME") or "").strip()
    if legacy:
        return legacy
    return _DEFAULT_DATA_HOME


def _data_home_path(data_home: str, *parts: str) -> str:
    return (Path(data_home) / Path(*parts)).as_posix()


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


def _parse_tenant_ids(raw: str) -> tuple[str, ...]:
    seen: set[str] = set()
    tenant_ids: list[str] = []
    for value in raw.split(","):
        tenant_id = value.strip()
        if tenant_id and tenant_id not in seen:
            seen.add(tenant_id)
            tenant_ids.append(tenant_id)
    return tuple(tenant_ids)


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
    observability_sentry_dsn: str = ""
    observability_sentry_environment: str = ""
    observability_sentry_release: str = ""
    observability_sentry_server_name: str = ""
    observability_sentry_shutdown_timeout_seconds: float = 2.0
    observability_sentry_debug: bool = False
    observability_sentry_flush_after_capture: bool = False
    data_home: str = _DEFAULT_DATA_HOME
    document_store_backend: Literal["auto", "mongodb", "inmemory"] = "auto"
    file_watcher_enabled: bool = False
    file_watcher_tenant_id: str = ""
    file_watcher_workspace_id: str = ""
    file_watcher_collection_id: str = ""
    file_watcher_poll_interval_seconds: float = 1.0
    file_watcher_debounce_seconds: float = 1.0
    file_watcher_max_batch_wait_seconds: float = 10.0
    file_watcher_priority: str = "normal"
    # Optional Slack Ask companion (LKW-SLACK-WORKFLOW-1A). Incomplete values when
    # enabled degrade only the companion — never core HTTP/MCP readiness.
    slack_companion_enabled: bool = False
    slack_approved_team_id: str = ""
    slack_approved_user_id: str = ""
    slack_tenant_id: str = ""
    slack_active_workspace_id: str = ""
    slack_ask_base_url: str = ""
    slack_ask_api_key: str = ""
    slack_ask_timeout_seconds: float = 60.0
    managed_file_max_bytes: int = 25 * 1024 * 1024
    managed_file_max_batch_files: int = 20
    web_url_preflight_timeout_seconds: float = 10.0
    connected_source_opaque_ref_signing_key: str = ""
    connected_source_slack_connection_ref: str = ""
    tenant_connection_bootstrap_tenant_ids: tuple[str, ...] = ()
    knowledge_admin_confirmation_secret: str = field(default="", repr=False)
    conversation_thread_memory_max_messages: int = 20
    conversation_thread_memory_max_bytes: int = 16 * 1024
    conversation_thread_memory_max_age_seconds: int = 24 * 60 * 60

    @property
    def config_dir(self) -> str:
        return _data_home_path(self.data_home, "config")

    @property
    def source_candidates_file(self) -> str:
        return _data_home_path(self.data_home, "config", "source_candidates.json")

    @property
    def data_dir(self) -> str:
        return _data_home_path(self.data_home, "data")

    @property
    def sqlite_data_dir(self) -> str:
        return _data_home_path(self.data_home, "data", "sqlite")

    @property
    def shadow_workspaces_dir(self) -> str:
        return _data_home_path(self.data_home, "data", "shadow_workspaces")

    @property
    def managed_file_storage_dir(self) -> str:
        return _data_home_path(self.data_home, "data", "managed_files")

    @property
    def managed_upload_staging_dir(self) -> str:
        return _data_home_path(self.data_home, "run", "managed_upload_staging")

    @property
    def web_url_staging_dir(self) -> str:
        return _data_home_path(self.data_home, "run", "web_url_staging")

    @property
    def logs_dir(self) -> str:
        return _data_home_path(self.data_home, "logs")

    @property
    def run_dir(self) -> str:
        return _data_home_path(self.data_home, "run")

    def validate_for_runtime(self) -> None:
        """Reject unsafe or incomplete production configuration before wiring."""
        if self.environment != ApiEnvironment.PROD:
            return

        if self.data_home.strip() in {"", _DEFAULT_DATA_HOME}:
            raise ValueError(
                "LOCAL_WORKSPACE_DATA_HOME is required for production durable storage."
            )
        if self.document_store_backend not in {"auto", "mongodb", "inmemory"}:
            raise ValueError("local_workspace_document_store_backend_invalid")
        if self.document_store_backend == "inmemory":
            raise ValueError(
                "LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND=inmemory is development-only."
            )
        if not (os.environ.get("INTERGRAX_MONGODB_URI") or "").strip():
            raise ValueError(
                "INTERGRAX_MONGODB_URI is required for production durable workspace state."
            )

        vector_store = (os.environ.get("LOCAL_WORKSPACE_VECTOR_STORE") or "qdrant").strip().lower()
        if vector_store == "inmemory":
            raise ValueError(
                "LOCAL_WORKSPACE_VECTOR_STORE=inmemory is development-only."
            )
        if vector_store == "qdrant" and not (
            os.environ.get("INTERGRAX_QDRANT_URL") or ""
        ).strip():
            raise ValueError(
                "INTERGRAX_QDRANT_URL is required for production indexed storage."
            )

        live_values = (
            self.connected_source_opaque_ref_signing_key.strip(),
            self.slack_tenant_id.strip(),
            self.connected_source_slack_connection_ref.strip(),
        )
        if any(live_values) and not all(live_values):
            raise ValueError(
                "connected_source_live_configuration_incomplete"
            )
        confirmation_secret = self.knowledge_admin_confirmation_secret.strip()
        if confirmation_secret and len(confirmation_secret) < 32:
            raise ValueError(
                "LOCAL_WORKSPACE_KNOWLEDGE_ADMIN_CONFIRMATION_SECRET is too short."
            )

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
    def build_observability_export_config(
        self,
    ) -> ObservabilityExportOperatorConfig | None:
        """Build optional ObservabilityExportOperatorConfig from env-driven settings.

        Returns None when export is disabled.  Raises ValueError on
        missing endpoint or unsupported backend.
        """
        if not self.observability_export_enabled:
            return None

        try:
            backend_id = parse_observability_export_backend_id(
                self.observability_export_backend
            )
        except ObservabilityExportOperatorConfigError as exc:
            raise ValueError(str(exc)) from exc

        otlp: OtlpExportOperatorConfig | None
        elasticsearch: ElasticsearchExportOperatorConfig | None
        sentry: SentryExportOperatorConfig | None
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
            sentry = None
        elif backend_id == "elasticsearch":
            otlp = None
            sentry = None
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
                self.observability_elasticsearch_failed_delivery_file_path.strip()
                or None
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
        elif backend_id == "sentry":
            otlp = None
            elasticsearch = None
            dsn = self.observability_sentry_dsn.strip()
            if not dsn:
                raise ValueError(
                    "LOCAL_WORKSPACE_OBSERVABILITY_SENTRY_DSN is required when "
                    "observability export is enabled with backend_id=sentry"
                )
            sentry = SentryExportOperatorConfig(
                dsn=dsn,
                environment=self.observability_sentry_environment or None,
                release=self.observability_sentry_release or None,
                server_name=self.observability_sentry_server_name or None,
                shutdown_timeout_seconds=self.observability_sentry_shutdown_timeout_seconds,
                debug=self.observability_sentry_debug,
                flush_after_capture=self.observability_sentry_flush_after_capture,
            )
        else:
            otlp = None
            elasticsearch = None
            sentry = None

        # Safety: never export raw content regardless of env value
        return ObservabilityExportOperatorConfig(
            enabled=True,
            export_content=False,
            backend_id=backend_id,
            otlp=otlp,
            elasticsearch=elasticsearch,
            sentry=sentry,
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
                "context_only"
                if environment == ApiEnvironment.PROD
                else "body_or_context"
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
            default=cls._field_default(
                "observability_elasticsearch_retry_max_attempts"
            ),  # type: ignore[arg-type]
        )
        observability_elasticsearch_retry_initial_backoff_seconds = env.float(
            "OBSERVABILITY_ELASTICSEARCH_RETRY_INITIAL_BACKOFF_SECONDS",
            default=cls._field_default(
                "observability_elasticsearch_retry_initial_backoff_seconds"
            ),  # type: ignore[arg-type]
        )
        observability_elasticsearch_retry_max_backoff_seconds = env.float(
            "OBSERVABILITY_ELASTICSEARCH_RETRY_MAX_BACKOFF_SECONDS",
            default=cls._field_default(
                "observability_elasticsearch_retry_max_backoff_seconds"
            ),  # type: ignore[arg-type]
        )
        observability_elasticsearch_failed_delivery_file_path = env.str(
            "OBSERVABILITY_ELASTICSEARCH_FAILED_DELIVERY_FILE_PATH",
            default=cls._field_default(
                "observability_elasticsearch_failed_delivery_file_path"
            ),  # type: ignore[arg-type]
        )
        observability_sentry_dsn = env.str(
            "OBSERVABILITY_SENTRY_DSN",
            default=cls._field_default("observability_sentry_dsn"),  # type: ignore[arg-type]
        )
        observability_sentry_environment = env.str(
            "OBSERVABILITY_SENTRY_ENVIRONMENT",
            default=cls._field_default("observability_sentry_environment"),  # type: ignore[arg-type]
        )
        observability_sentry_release = env.str(
            "OBSERVABILITY_SENTRY_RELEASE",
            default=cls._field_default("observability_sentry_release"),  # type: ignore[arg-type]
        )
        observability_sentry_server_name = env.str(
            "OBSERVABILITY_SENTRY_SERVER_NAME",
            default=cls._field_default("observability_sentry_server_name"),  # type: ignore[arg-type]
        )
        observability_sentry_shutdown_timeout_seconds = env.float(
            "OBSERVABILITY_SENTRY_SHUTDOWN_TIMEOUT_SECONDS",
            default=cls._field_default("observability_sentry_shutdown_timeout_seconds"),  # type: ignore[arg-type]
        )
        observability_sentry_debug = env.bool(
            "OBSERVABILITY_SENTRY_DEBUG",
            default=cls._field_default("observability_sentry_debug"),  # type: ignore[arg-type]
        )
        observability_sentry_flush_after_capture = env.bool(
            "OBSERVABILITY_SENTRY_FLUSH_AFTER_CAPTURE",
            default=cls._field_default("observability_sentry_flush_after_capture"),  # type: ignore[arg-type]
        )
        read_roots_raw = (os.environ.get("INTERGRAX_ALLOWED_READ_ROOTS") or "").strip()
        allowed_read_roots = frozenset(
            part.strip() for part in read_roots_raw.split(",") if part.strip()
        )
        data_home = _resolve_data_home(env)
        file_watcher_enabled = env.bool(
            "FILE_WATCHER_ENABLED",
            default=cls._field_default("file_watcher_enabled"),  # type: ignore[arg-type]
        )
        file_watcher_tenant_id = env.str(
            "FILE_WATCHER_TENANT_ID",
            default=cls._field_default("file_watcher_tenant_id"),  # type: ignore[arg-type]
        )
        file_watcher_workspace_id = env.str(
            "FILE_WATCHER_WORKSPACE_ID",
            default=cls._field_default("file_watcher_workspace_id"),  # type: ignore[arg-type]
        )
        file_watcher_collection_id = env.str(
            "FILE_WATCHER_COLLECTION_ID",
            default=cls._field_default("file_watcher_collection_id"),  # type: ignore[arg-type]
        )
        file_watcher_poll_interval_seconds = env.float(
            "FILE_WATCHER_POLL_INTERVAL_SECONDS",
            default=cls._field_default("file_watcher_poll_interval_seconds"),  # type: ignore[arg-type]
        )
        file_watcher_debounce_seconds = env.float(
            "FILE_WATCHER_DEBOUNCE_SECONDS",
            default=cls._field_default("file_watcher_debounce_seconds"),  # type: ignore[arg-type]
        )
        file_watcher_max_batch_wait_seconds = env.float(
            "FILE_WATCHER_MAX_BATCH_WAIT_SECONDS",
            default=cls._field_default("file_watcher_max_batch_wait_seconds"),  # type: ignore[arg-type]
        )
        file_watcher_priority = env.str(
            "FILE_WATCHER_PRIORITY",
            default=cls._field_default("file_watcher_priority"),  # type: ignore[arg-type]
        )
        slack_companion_enabled = env.bool(
            "SLACK_COMPANION_ENABLED",
            default=cls._field_default("slack_companion_enabled"),  # type: ignore[arg-type]
        )
        slack_approved_team_id = env.str(
            "SLACK_APPROVED_TEAM_ID",
            default=cls._field_default("slack_approved_team_id"),  # type: ignore[arg-type]
        )
        slack_approved_user_id = env.str(
            "SLACK_APPROVED_USER_ID",
            default=cls._field_default("slack_approved_user_id"),  # type: ignore[arg-type]
        )
        slack_tenant_id = env.str(
            "SLACK_TENANT_ID",
            default=cls._field_default("slack_tenant_id"),  # type: ignore[arg-type]
        )
        slack_active_workspace_id = env.str(
            "SLACK_ACTIVE_WORKSPACE_ID",
            default=cls._field_default("slack_active_workspace_id"),  # type: ignore[arg-type]
        )
        slack_ask_base_url = env.str(
            "SLACK_ASK_BASE_URL",
            default=cls._field_default("slack_ask_base_url"),  # type: ignore[arg-type]
        )
        slack_ask_api_key = env.str(
            "SLACK_ASK_API_KEY",
            default=cls._field_default("slack_ask_api_key"),  # type: ignore[arg-type]
        )
        slack_ask_timeout_seconds = env.float(
            "SLACK_ASK_TIMEOUT_SECONDS",
            default=cls._field_default("slack_ask_timeout_seconds"),  # type: ignore[arg-type]
        )
        managed_file_max_bytes = env.int(
            "MANAGED_FILE_MAX_BYTES",
            default=cls._field_default("managed_file_max_bytes"),  # type: ignore[arg-type]
        )
        managed_file_max_batch_files = env.int(
            "MANAGED_FILE_MAX_BATCH_FILES",
            default=cls._field_default("managed_file_max_batch_files"),  # type: ignore[arg-type]
        )
        if managed_file_max_bytes < 1:
            raise ValueError("LOCAL_WORKSPACE_MANAGED_FILE_MAX_BYTES must be >= 1")
        if managed_file_max_batch_files < 1:
            raise ValueError("LOCAL_WORKSPACE_MANAGED_FILE_MAX_BATCH_FILES must be >= 1")

        connected_source_opaque_ref_signing_key = env.str(
            "CONNECTED_SOURCE_OPAQUE_REF_SIGNING_KEY",
            default=cls._field_default("connected_source_opaque_ref_signing_key"),  # type: ignore[arg-type]
        )
        connected_source_slack_connection_ref = env.str(
            "CONNECTED_SOURCE_SLACK_CONNECTION_REF",
            default=cls._field_default("connected_source_slack_connection_ref"),  # type: ignore[arg-type]
        )
        tenant_connection_bootstrap_tenant_ids = _parse_tenant_ids(
            env.str(
                "TENANT_CONNECTION_BOOTSTRAP_TENANT_IDS",
                default="",
            )
        )
        knowledge_admin_confirmation_secret = env.str(
            "KNOWLEDGE_ADMIN_CONFIRMATION_SECRET",
            default=cls._field_default("knowledge_admin_confirmation_secret"),  # type: ignore[arg-type]
        )
        document_store_backend = (
            env.str("DOCUMENT_STORE_BACKEND", default="auto").strip().lower() or "auto"
        )
        if document_store_backend not in {"auto", "mongodb", "inmemory"}:
            raise ValueError(
                "LOCAL_WORKSPACE_DOCUMENT_STORE_BACKEND must be one of: auto, mongodb, inmemory."
            )

        managed_staging_root = _data_home_path(data_home, "run", "managed_upload_staging")
        web_url_staging_root = _data_home_path(data_home, "run", "web_url_staging")
        allowed_read_roots = frozenset(
            set(allowed_read_roots) | {managed_staging_root, web_url_staging_root}
        )

        return {
            "data_home": data_home,
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
            "observability_sentry_dsn": observability_sentry_dsn,
            "observability_sentry_environment": observability_sentry_environment,
            "observability_sentry_release": observability_sentry_release,
            "observability_sentry_server_name": observability_sentry_server_name,
            "observability_sentry_shutdown_timeout_seconds": observability_sentry_shutdown_timeout_seconds,
            "observability_sentry_debug": observability_sentry_debug,
            "observability_sentry_flush_after_capture": observability_sentry_flush_after_capture,
            "document_store_backend": document_store_backend,
            "file_watcher_enabled": file_watcher_enabled,
            "file_watcher_tenant_id": file_watcher_tenant_id,
            "file_watcher_workspace_id": file_watcher_workspace_id,
            "file_watcher_collection_id": file_watcher_collection_id,
            "file_watcher_poll_interval_seconds": file_watcher_poll_interval_seconds,
            "file_watcher_debounce_seconds": file_watcher_debounce_seconds,
            "file_watcher_max_batch_wait_seconds": file_watcher_max_batch_wait_seconds,
            "file_watcher_priority": file_watcher_priority,
            "slack_companion_enabled": slack_companion_enabled,
            "slack_approved_team_id": slack_approved_team_id,
            "slack_approved_user_id": slack_approved_user_id,
            "slack_tenant_id": slack_tenant_id,
            "slack_active_workspace_id": slack_active_workspace_id,
            "slack_ask_base_url": slack_ask_base_url,
            "slack_ask_api_key": slack_ask_api_key,
            "slack_ask_timeout_seconds": slack_ask_timeout_seconds,
            "managed_file_max_bytes": managed_file_max_bytes,
            "managed_file_max_batch_files": managed_file_max_batch_files,
            "connected_source_opaque_ref_signing_key": connected_source_opaque_ref_signing_key,
            "connected_source_slack_connection_ref": connected_source_slack_connection_ref,
            "tenant_connection_bootstrap_tenant_ids": tenant_connection_bootstrap_tenant_ids,
            "knowledge_admin_confirmation_secret": knowledge_admin_confirmation_secret,
        }
