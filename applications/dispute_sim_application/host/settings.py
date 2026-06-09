# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Mapping, Optional

from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment

DisputeSimIdentitySource = Literal["body_or_context", "context_only"]


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
        raise ValueError("DISPUTE_SIM_BACKEND_API_KEYS_JSON must be a JSON object.")
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
class DisputeSimBackendSettings:
    """Environment for dispute_sim_application (scaffolded product profile)."""

    environment: ApiEnvironment
    route_prefix: str = "/v1/dispute_sim"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8025
    default_agent_id: str = "dispute_intake"
    identity_source: DisputeSimIdentitySource = "body_or_context"
    cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
    openapi_enabled_override: Optional[bool] = None
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
    include_mcp: bool = True
    mcp_mount_path: str = "/mcp"
    include_task_control: bool = True
    include_scheduler: bool = False
    task_control_route_prefix: str = "/v1/tasks"
    scheduler_poll_seconds: float | None = None

    @classmethod
    def from_env(cls) -> DisputeSimBackendSettings:
        env_raw = (os.getenv("DISPUTE_SIM_BACKEND_ENV") or os.getenv("INTERGRAX_ENV") or "dev").strip().lower()
        try:
            environment = ApiEnvironment(env_raw)
        except ValueError as exc:
            raise ValueError(
                f"DISPUTE_SIM_BACKEND_ENV must be one of "
                f"{[e.value for e in ApiEnvironment]}, got {env_raw!r}."
            ) from exc

        prefix = (os.getenv("DISPUTE_SIM_ROUTE_PREFIX") or "/v1/dispute_sim").strip() or "/v1/dispute_sim"
        host = (os.getenv("DISPUTE_SIM_BACKEND_HOST") or "127.0.0.1").strip()
        port_raw = (os.getenv("DISPUTE_SIM_BACKEND_PORT") or "8025").strip()
        agent_id = (os.getenv("DISPUTE_SIM_DEFAULT_AGENT_ID") or "dispute_intake").strip() or "dispute_intake"

        id_src_env = os.getenv("DISPUTE_SIM_IDENTITY_SOURCE", "").strip().lower()
        if id_src_env in {"body_or_context", "context_only"}:
            identity_source = id_src_env
        else:
            identity_source = "context_only" if environment == ApiEnvironment.PROD else "body_or_context"

        cors = _env_csv_set("DISPUTE_SIM_BACKEND_CORS_ORIGINS")
        hosts = _env_csv_set("DISPUTE_SIM_BACKEND_ALLOWED_HOSTS")

        openapi_override: Optional[bool] = None
        if os.getenv("DISPUTE_SIM_BACKEND_OPENAPI") is not None:
            openapi_override = _env_bool("DISPUTE_SIM_BACKEND_OPENAPI")

        keys: Mapping[str, ApiKeyIdentity] = {}
        bootstrap_key = os.getenv("DISPUTE_SIM_BACKEND_BOOTSTRAP_API_KEY", "").strip()
        if bootstrap_key:
            tenant = os.getenv("DISPUTE_SIM_BACKEND_BOOTSTRAP_TENANT_ID", "").strip()
            user = os.getenv("DISPUTE_SIM_BACKEND_BOOTSTRAP_USER_ID", "").strip()
            if not tenant or not user:
                raise ValueError(
                    "When DISPUTE_SIM_BACKEND_BOOTSTRAP_API_KEY is set, "
                    "DISPUTE_SIM_BACKEND_BOOTSTRAP_TENANT_ID and "
                    "DISPUTE_SIM_BACKEND_BOOTSTRAP_USER_ID are required."
                )
            keys = {
                bootstrap_key: ApiKeyIdentity(
                    tenant_id=tenant,
                    user_id=user,
                    scopes=("*",),
                )
            }
        json_keys = os.getenv("DISPUTE_SIM_BACKEND_API_KEYS_JSON", "").strip()
        if json_keys:
            if keys:
                raise ValueError(
                    "Use either DISPUTE_SIM_BACKEND_BOOTSTRAP_API_KEY or "
                    "DISPUTE_SIM_BACKEND_API_KEYS_JSON, not both."
                )
            keys = _parse_api_key_map(json_keys)

        if environment == ApiEnvironment.PROD and identity_source != "context_only":
            raise ValueError(
                "DISPUTE_SIM_BACKEND_ENV=prod requires "
                "DISPUTE_SIM_IDENTITY_SOURCE=context_only (or omit to default)."
            )
        if environment == ApiEnvironment.PROD and not _env_bool(
            "DISPUTE_SIM_BACKEND_ALLOW_UNAUTHENTICATED", False
        ):
            if not keys:
                raise ValueError(
                    "Production dispute_sim backend requires API keys: set "
                    "DISPUTE_SIM_BACKEND_BOOTSTRAP_API_KEY (+ tenant/user) or "
                    "DISPUTE_SIM_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set "
                    "DISPUTE_SIM_BACKEND_ALLOW_UNAUTHENTICATED=true."
                )

        include_mcp = _env_bool("DISPUTE_SIM_INCLUDE_MCP", default=True)
        mcp_mount = (os.getenv("DISPUTE_SIM_MCP_MOUNT_PATH") or "/mcp").strip() or "/mcp"
        include_task_control = _env_bool("DISPUTE_SIM_INCLUDE_TASK_CONTROL", default=True)
        include_scheduler = _env_bool("DISPUTE_SIM_INCLUDE_SCHEDULER", default=False)
        task_control_prefix = (
            os.getenv("DISPUTE_SIM_TASK_CONTROL_ROUTE_PREFIX") or "/v1/tasks"
        ).strip() or "/v1/tasks"
        poll_raw = (os.getenv("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None

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
            include_task_control=include_task_control,
            include_scheduler=include_scheduler,
            task_control_route_prefix=task_control_prefix,
            scheduler_poll_seconds=scheduler_poll,
        )
