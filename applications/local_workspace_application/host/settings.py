# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Mapping, Optional

from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment

LocalWorkspaceIdentitySource = Literal["body_or_context", "context_only"]


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


@dataclass(frozen=True)
class LocalWorkspaceBackendSettings:
    """Environment for local_workspace_application (scaffolded product profile)."""

    environment: ApiEnvironment
    route_prefix: str = "/v1/local_workspace"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8020
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
    include_mcp: bool = True
    mcp_mount_path: str = "/mcp"

    @property
    def enabled_tool_ids(self) -> list[str]:
        """Catalog tool_ids enabled for this host (from env flags)."""
        ids: list[str] = list(self.extra_enabled_tool_ids)
        if self.enable_rag and "rag.retrieve" not in ids:
            ids.append("rag.retrieve")
        if self.enable_rag_ingest and "rag.ingest_document" not in ids:
            ids.append("rag.ingest_document")
        return ids

    @classmethod
    def from_env(cls) -> LocalWorkspaceBackendSettings:
        env_raw = (os.getenv("LOCAL_WORKSPACE_BACKEND_ENV") or os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
        try:
            environment = ApiEnvironment(env_raw)
        except ValueError as exc:
            raise ValueError(
                f"LOCAL_WORKSPACE_BACKEND_ENV must be one of "
                f"{[e.value for e in ApiEnvironment]}, got {env_raw!r}."
            ) from exc

        prefix = (os.getenv("LOCAL_WORKSPACE_ROUTE_PREFIX") or "/v1/local_workspace").strip() or "/v1/local_workspace"
        host = (os.getenv("LOCAL_WORKSPACE_BACKEND_HOST") or "127.0.0.1").strip()
        port_raw = (os.getenv("LOCAL_WORKSPACE_BACKEND_PORT") or "8020").strip()
        agent_id = (os.getenv("LOCAL_WORKSPACE_DEFAULT_AGENT_ID") or "local_search").strip() or "local_search"
        enable_rag = _env_bool("LOCAL_WORKSPACE_ENABLE_RAG", default=True)
        enable_rag_ingest = _env_bool("LOCAL_WORKSPACE_ENABLE_RAG_INGEST", default=True)

        id_src_env = os.getenv("LOCAL_WORKSPACE_IDENTITY_SOURCE", "").strip().lower()
        if id_src_env in {"body_or_context", "context_only"}:
            identity_source = id_src_env
        else:
            identity_source = "context_only" if environment == ApiEnvironment.PROD else "body_or_context"

        cors = _env_csv_set("LOCAL_WORKSPACE_BACKEND_CORS_ORIGINS")
        hosts = _env_csv_set("LOCAL_WORKSPACE_BACKEND_ALLOWED_HOSTS")

        openapi_override: Optional[bool] = None
        if os.getenv("LOCAL_WORKSPACE_BACKEND_OPENAPI") is not None:
            openapi_override = _env_bool("LOCAL_WORKSPACE_BACKEND_OPENAPI")

        keys: Mapping[str, ApiKeyIdentity] = {}
        bootstrap_key = os.getenv("LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY", "").strip()
        if bootstrap_key:
            tenant = os.getenv("LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_TENANT_ID", "").strip()
            user = os.getenv("LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_USER_ID", "").strip()
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
        json_keys = os.getenv("LOCAL_WORKSPACE_BACKEND_API_KEYS_JSON", "").strip()
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
        if environment == ApiEnvironment.PROD and not _env_bool(
            "LOCAL_WORKSPACE_BACKEND_ALLOW_UNAUTHENTICATED", False
        ):
            if not keys:
                raise ValueError(
                    "Production local_workspace backend requires API keys: set "
                    "LOCAL_WORKSPACE_BACKEND_BOOTSTRAP_API_KEY (+ tenant/user) or "
                    "LOCAL_WORKSPACE_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set "
                    "LOCAL_WORKSPACE_BACKEND_ALLOW_UNAUTHENTICATED=true."
                )

        include_mcp = _env_bool("LOCAL_WORKSPACE_INCLUDE_MCP", default=True)
        mcp_mount = (os.getenv("LOCAL_WORKSPACE_MCP_MOUNT_PATH") or "/mcp").strip() or "/mcp"

        return cls(
            environment=environment,
            route_prefix=prefix,
            backend_host=host,
            backend_port=int(port_raw),
            default_agent_id=agent_id,
            identity_source=identity_source,
            cors_allow_origins=cors,
            allowed_hosts=hosts,
            openapi_enabled_override=openapi_override,
            api_keys_map=keys,
            include_mcp=include_mcp,
            mcp_mount_path=mcp_mount,
            enable_rag=enable_rag,
            enable_rag_ingest=enable_rag_ingest,
            allowed_read_roots=_env_csv_set("INTERGRAX_ALLOWED_READ_ROOTS"),
        )
