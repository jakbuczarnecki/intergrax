# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import ClassVar, FrozenSet, Literal, Mapping, Optional

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment

GovernedContractorIdentitySource = Literal["body_or_context", "context_only"]


def _parse_api_key_map(raw: Optional[str]) -> Mapping[str, ApiKeyIdentity]:
    if not raw or not raw.strip():
        return {}
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON must be a JSON object.")
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
class GovernedContractorBackendSettings(IntergraxApplicationSettingsBase):
    """Environment for governed_contractor_application (scaffolded product profile)."""

    env_prefix: ClassVar[str] = "GOVERNED_CONTRACTOR_"
    route_prefix: str = "/v1/governed_contractor"
    backend_port: int = 8000
    include_scheduler: bool = False
    include_queue_worker: bool = False
    default_agent_id: str = "external_contractor_adapter"
    identity_source: GovernedContractorIdentitySource = "body_or_context"
    cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
    openapi_enabled_override: Optional[bool] = None
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
    interaction_execute_default: bool = True

    # Programmatic DI slots (not env-backed) — Execution Evidence / GEC wiring.
    # Set on a settings instance or build-context settings object before mount.
    external_work_integration: object | None = None
    meaningful_side_effect_authorization_boundary: object | None = None
    runtime_policy_bundle: object | None = None
    host_attestor: object | None = None
    attestation_required: bool = False

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
        environment = ApiEnvironment(env_raw)

        agent_id = (
            env.str("DEFAULT_AGENT_ID", default="external_contractor_adapter")
            or "external_contractor_adapter"
        )

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
                    "When GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_API_KEY is set, "
                    "GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_TENANT_ID and "
                    "GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_USER_ID are required."
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
                    "Use either GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_API_KEY or "
                    "GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON, not both."
                )
            keys = _parse_api_key_map(json_keys)

        if environment == ApiEnvironment.PROD and identity_source != "context_only":
            raise ValueError(
                "GOVERNED_CONTRACTOR_BACKEND_ENV=prod requires "
                "GOVERNED_CONTRACTOR_IDENTITY_SOURCE=context_only (or omit to default)."
            )
        if environment == ApiEnvironment.PROD and not env.bool(
            "BACKEND_ALLOW_UNAUTHENTICATED", default=False
        ):
            if not keys:
                raise ValueError(
                    "Production governed_contractor backend requires API keys: set "
                    "GOVERNED_CONTRACTOR_BACKEND_BOOTSTRAP_API_KEY (+ tenant/user) or "
                    "GOVERNED_CONTRACTOR_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set "
                    "GOVERNED_CONTRACTOR_BACKEND_ALLOW_UNAUTHENTICATED=true."
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
