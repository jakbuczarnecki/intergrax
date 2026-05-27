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
    use_nexus_loop: bool
    cors_allow_origins: FrozenSet[str]
    allowed_hosts: FrozenSet[str]
    openapi_enabled_override: Optional[bool]
    session_sqlite_path: Optional[str]
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)

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
        agent_id = os.environ.get("LEGAL_DEFAULT_AGENT_ID", "legal-default").strip()
        prefix = os.environ.get("LEGAL_ROUTE_PREFIX", "/v1/legal").strip() or "/v1/legal"
        use_nexus_loop = _env_bool("LEGAL_USE_NEXUS_LOOP", environment == ApiEnvironment.DEV)

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

        return cls(
            environment=environment,
            legal_product_profile=profile,
            legal_llm_provider=llm,
            legal_default_agent_id=agent_id,
            legal_route_prefix=prefix,
            identity_source=identity_source,
            use_nexus_loop=use_nexus_loop,
            cors_allow_origins=cors,
            allowed_hosts=hosts,
            openapi_enabled_override=openapi_override,
            session_sqlite_path=session_db,
            api_keys_map=keys,
        )
