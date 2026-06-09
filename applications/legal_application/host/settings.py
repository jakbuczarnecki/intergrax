# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""
Environment-driven settings for the Legal backend host.

Fail-fast on invalid combinations (e.g. PROD without authentication).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Mapping, Optional

from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment

LegalIdentitySource = Literal["body_or_context", "context_only"]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_csv_set(name: str) -> FrozenSet[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return frozenset()
    return frozenset(x.strip() for x in raw.split(",") if x.strip())


def _parse_api_key_map(raw: Optional[str]) -> Mapping[str, ApiKeyIdentity]:
    """
    JSON object: { \"api_key_string\": { \"tenant_id\": \"...\", \"user_id\": \"...\", \"scopes\": [\"*\"] } }
    """
    if not raw or not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("LEGAL_BACKEND_API_KEYS_JSON must be a JSON object.")
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


@dataclass(frozen=True)
class LegalBackendSettings:
    """Loaded once at process start from environment variables."""

    environment: ApiEnvironment
    legal_product_profile: str
    legal_llm_provider: str
    legal_default_agent_id: str
    legal_route_prefix: str
    identity_source: LegalIdentitySource
    cors_allow_origins: FrozenSet[str]
    allowed_hosts: FrozenSet[str]
    openapi_enabled_override: Optional[bool]
    session_sqlite_path: Optional[str]
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
    include_mcp: bool = True
    mcp_mount_path: str = "/mcp"
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    interaction_surface: str = "auto"
    interaction_execute_default: bool = True
    include_task_control: bool = True
    include_scheduler: bool = False
    include_queue_worker: bool = False
    task_control_route_prefix: str = "/v1/tasks"
    scheduler_poll_seconds: float | None = None
    enable_rag: bool = False
    enable_rag_ingest: bool = False
    enable_websearch: bool = False
    use_legal_tool_decision: bool = False
    tools_mode: str = "off"
    extra_enabled_tool_ids: tuple[str, ...] = ()
    enable_modality_tools: bool = False
    legal_llm_model: Optional[str] = None

    @property
    def enabled_tool_ids(self) -> list[str]:
        """Catalog tool_ids enabled for this host (from env flags)."""
        ids: list[str] = list(self.extra_enabled_tool_ids)
        if self.enable_rag and "rag.retrieve" not in ids:
            ids.append("rag.retrieve")
        if self.enable_rag_ingest and "rag.ingest_document" not in ids:
            ids.append("rag.ingest_document")
        if self.enable_websearch and "websearch.query" not in ids:
            ids.append("websearch.query")
        return ids

    @classmethod
    def from_env(cls) -> LegalBackendSettings:
        env_raw = os.environ.get("LEGAL_BACKEND_ENV", "dev").strip().lower()
        try:
            environment = ApiEnvironment(env_raw)
        except ValueError as exc:
            raise ValueError(
                f"LEGAL_BACKEND_ENV must be one of {[e.value for e in ApiEnvironment]}, got {env_raw!r}."
            ) from exc

        profile = os.environ.get("LEGAL_PRODUCT_PROFILE", "strict_legal").strip()
        llm = os.environ.get("LEGAL_LLM_PROVIDER", "ollama").strip().lower()
        llm_model = os.environ.get("LEGAL_LLM_MODEL", "").strip() or None
        agent_id = os.environ.get("LEGAL_DEFAULT_AGENT_ID", "legal-default").strip()
        prefix = os.environ.get("LEGAL_ROUTE_PREFIX", "/v1/legal").strip() or "/v1/legal"

        id_src_env = os.environ.get("LEGAL_IDENTITY_SOURCE", "").strip().lower()
        if id_src_env in {"body_or_context", "context_only"}:
            identity_source = id_src_env
        else:
            identity_source = "context_only" if environment == ApiEnvironment.PROD else "body_or_context"

        cors = _env_csv_set("LEGAL_BACKEND_CORS_ORIGINS")
        hosts = _env_csv_set("LEGAL_BACKEND_ALLOWED_HOSTS")

        openapi_override: Optional[bool] = None
        if os.environ.get("LEGAL_BACKEND_OPENAPI") is not None:
            openapi_override = _env_bool("LEGAL_BACKEND_OPENAPI")

        session_db = os.environ.get("LEGAL_SESSION_SQLITE_PATH", "").strip() or None

        keys: Mapping[str, ApiKeyIdentity] = {}
        bootstrap_key = os.environ.get("LEGAL_BACKEND_BOOTSTRAP_API_KEY", "").strip()
        if bootstrap_key:
            tenant = os.environ.get("LEGAL_BACKEND_BOOTSTRAP_TENANT_ID", "").strip()
            user = os.environ.get("LEGAL_BACKEND_BOOTSTRAP_USER_ID", "").strip()
            if not tenant or not user:
                raise ValueError(
                    "When LEGAL_BACKEND_BOOTSTRAP_API_KEY is set, "
                    "LEGAL_BACKEND_BOOTSTRAP_TENANT_ID and LEGAL_BACKEND_BOOTSTRAP_USER_ID are required."
                )
            keys = {
                bootstrap_key: ApiKeyIdentity(
                    tenant_id=tenant,
                    user_id=user,
                    scopes=("*",),
                )
            }
        json_keys = os.environ.get("LEGAL_BACKEND_API_KEYS_JSON", "").strip()
        if json_keys:
            if keys:
                raise ValueError("Use either LEGAL_BACKEND_BOOTSTRAP_API_KEY or LEGAL_BACKEND_API_KEYS_JSON, not both.")
            keys = _parse_api_key_map(json_keys)

        if environment == ApiEnvironment.PROD and identity_source != "context_only":
            raise ValueError("LEGAL_BACKEND_ENV=prod requires LEGAL_IDENTITY_SOURCE=context_only (or omit to default).")
        if environment == ApiEnvironment.PROD and not _env_bool("LEGAL_BACKEND_ALLOW_UNAUTHENTICATED", False):
            if not keys:
                raise ValueError(
                    "Production Legal backend requires API keys: set LEGAL_BACKEND_BOOTSTRAP_API_KEY "
                    "(+ tenant/user) or LEGAL_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set LEGAL_BACKEND_ALLOW_UNAUTHENTICATED=true."
                )

        include_mcp = _env_bool("LEGAL_INCLUDE_MCP", default=True)
        mcp_mount = os.environ.get("LEGAL_MCP_MOUNT_PATH", "/mcp").strip() or "/mcp"
        include_interactions = _env_bool("LEGAL_INCLUDE_INTERACTIONS", default=True)
        interaction_prefix = (
            os.environ.get("LEGAL_INTERACTION_ROUTE_PREFIX") or "/v1/interactions"
        ).strip() or "/v1/interactions"
        interaction_surface = (
            os.environ.get("LEGAL_INTERACTION_SURFACE") or "auto"
        ).strip().lower() or "auto"
        interaction_execute = _env_bool("LEGAL_INTERACTION_EXECUTE_DEFAULT", default=True)
        include_task_control = _env_bool("LEGAL_INCLUDE_TASK_CONTROL", default=True)
        include_scheduler = _env_bool("LEGAL_INCLUDE_SCHEDULER", default=False)
        include_queue_worker = _env_bool("LEGAL_INCLUDE_QUEUE_WORKER", default=False)
        task_control_prefix = (
            os.environ.get("LEGAL_TASK_CONTROL_ROUTE_PREFIX") or "/v1/tasks"
        ).strip() or "/v1/tasks"
        poll_raw = (os.environ.get("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None

        enable_rag = _env_bool("LEGAL_ENABLE_RAG", default=False)
        enable_rag_ingest = _env_bool("LEGAL_ENABLE_RAG_INGEST", default=False)
        enable_websearch = _env_bool("LEGAL_ENABLE_WEBSEARCH", default=False)
        use_legal_tool_decision = _env_bool("LEGAL_USE_TOOL_DECISION", default=False)
        tools_mode = os.environ.get("LEGAL_TOOLS_MODE", "off").strip().lower() or "off"
        extra_tools_raw = os.environ.get("LEGAL_ENABLED_TOOLS", "").strip()
        extra_tools = tuple(x.strip() for x in extra_tools_raw.split(",") if x.strip())
        enable_modality = _env_bool("LEGAL_ENABLE_MODALITY_TOOLS", default=False)

        # Research SKU defaults — tools opt-in via env unless explicitly enabled
        if profile == "research" and not _env_bool("LEGAL_ENABLE_RAG", default=False) and os.environ.get("LEGAL_ENABLE_RAG") is None:
            enable_rag = True
        if profile == "research" and not _env_bool("LEGAL_ENABLE_WEBSEARCH", default=False) and os.environ.get("LEGAL_ENABLE_WEBSEARCH") is None:
            enable_websearch = True
        if profile == "research" and os.environ.get("LEGAL_USE_TOOL_DECISION") is None:
            use_legal_tool_decision = True

        return cls(
            environment=environment,
            legal_product_profile=profile,
            legal_llm_provider=llm,
            legal_llm_model=llm_model,
            legal_default_agent_id=agent_id,
            legal_route_prefix=prefix,
            identity_source=identity_source,
            cors_allow_origins=cors,
            allowed_hosts=hosts,
            openapi_enabled_override=openapi_override,
            session_sqlite_path=session_db,
            api_keys_map=keys,
            include_mcp=include_mcp,
            mcp_mount_path=mcp_mount,
            include_interaction_routes=include_interactions,
            interaction_route_prefix=interaction_prefix,
            interaction_surface=interaction_surface,
            interaction_execute_default=interaction_execute,
            include_task_control=include_task_control,
            include_scheduler=include_scheduler,
            include_queue_worker=include_queue_worker,
            task_control_route_prefix=task_control_prefix,
            scheduler_poll_seconds=scheduler_poll,
            enable_rag=enable_rag,
            enable_rag_ingest=enable_rag_ingest,
            enable_websearch=enable_websearch,
            use_legal_tool_decision=use_legal_tool_decision,
            tools_mode=tools_mode,
            extra_enabled_tool_ids=extra_tools,
            enable_modality_tools=enable_modality,
        )
