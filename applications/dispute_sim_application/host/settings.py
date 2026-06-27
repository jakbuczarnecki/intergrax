# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import ClassVar, FrozenSet, Literal, Mapping, Optional

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment

DisputeSimIdentitySource = Literal["body_or_context", "context_only"]


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


@dataclass(frozen=True, kw_only=True)
class DisputeSimBackendSettings(IntergraxApplicationSettingsBase):
    """Environment for dispute_sim_application (scaffolded product profile)."""

    env_prefix: ClassVar[str] = "DISPUTE_SIM_"
    route_prefix: str = "/v1/dispute_sim"
    backend_port: int = 8025
    include_scheduler: bool = True
    include_queue_worker: bool = False
    default_agent_id: str = "dispute_intake"
    identity_source: DisputeSimIdentitySource = "body_or_context"
    cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
    openapi_enabled_override: Optional[bool] = None
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
    interaction_execute_default: bool = True

    # ------------------------------------------------------------------
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

        agent_id = env.str("DEFAULT_AGENT_ID", default="dispute_intake") or "dispute_intake"

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
        json_keys = env.str("BACKEND_API_KEYS_JSON", default="")
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
        if environment == ApiEnvironment.PROD and not env.bool(
            "BACKEND_ALLOW_UNAUTHENTICATED", default=False
        ):
            if not keys:
                raise ValueError(
                    "Production dispute_sim backend requires API keys: set "
                    "DISPUTE_SIM_BACKEND_BOOTSTRAP_API_KEY (+ tenant/user) or "
                    "DISPUTE_SIM_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set "
                    "DISPUTE_SIM_BACKEND_ALLOW_UNAUTHENTICATED=true."
                )

        return {
            "default_agent_id": agent_id,
            "identity_source": identity_source,
            "cors_allow_origins": cors,
            "allowed_hosts": hosts,
            "openapi_enabled_override": openapi_override,
            "api_keys_map": keys,
            "interaction_execute_default": env.bool(
                "INTERACTION_EXECUTE_DEFAULT",
                default=cls._field_default("interaction_execute_default"),  # type: ignore[arg-type]
            ),
        }
