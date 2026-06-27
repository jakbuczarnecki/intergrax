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
from typing import ClassVar, FrozenSet, Literal, Mapping, Optional

from intergrax.applications.contracts.settings import EnvReader, IntergraxApplicationSettingsBase
from intergrax.fastapi_core.auth.api_key import ApiKeyIdentity
from intergrax.fastapi_core.config import ApiEnvironment

LegalIdentitySource = Literal["body_or_context", "context_only"]


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


@dataclass(frozen=True, kw_only=True)
class LegalBackendSettings(IntergraxApplicationSettingsBase):
    """Loaded once at process start from environment variables."""

    env_prefix: ClassVar[str] = "LEGAL_"
    route_prefix: str = "/v1/legal"
    include_scheduler: bool = True
    include_queue_worker: bool = False
    legal_product_profile: str = "strict_legal"
    legal_llm_provider: str = "ollama"
    legal_default_agent_id: str = "legal-default"
    identity_source: LegalIdentitySource = "body_or_context"
    cors_allow_origins: FrozenSet[str] = field(default_factory=frozenset)
    allowed_hosts: FrozenSet[str] = field(default_factory=frozenset)
    openapi_enabled_override: Optional[bool] = None
    session_sqlite_path: Optional[str] = None
    api_keys_map: Mapping[str, ApiKeyIdentity] = field(default_factory=dict)
    interaction_execute_default: bool = True
    enable_rag: bool = False
    enable_rag_ingest: bool = False
    enable_websearch: bool = False
    use_legal_tool_decision: bool = False
    tools_mode: str = "off"
    extra_enabled_tool_ids: tuple[str, ...] = ()
    enable_modality_tools: bool = False
    legal_llm_model: Optional[str] = None
    enable_llm_guardrails: bool = False
    llm_guardrail_primary: str = "llm_guard"
    llm_guardrail_semantic: str = "presidio"

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

    # ------------------------------------------------------------------
    # Application-specific settings
    # Add your own env-backed fields here.
    # ------------------------------------------------------------------

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        env_raw = env.optional_str("BACKEND_ENV") or (os.environ.get("INTERGRAX_ENV") or "dev").strip().lower()
        if env_raw == "staging":
            env_raw = "stage"
        environment = ApiEnvironment(env_raw)

        profile = env.str("PRODUCT_PROFILE", default="strict_legal")
        llm = env.str("LLM_PROVIDER", default="ollama").lower()
        llm_model = env.optional_str("LLM_MODEL")
        agent_id = env.str("DEFAULT_AGENT_ID", default="legal-default")

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

        session_db = env.optional_str("SESSION_SQLITE_PATH")

        keys: Mapping[str, ApiKeyIdentity] = {}
        bootstrap_key = env.str("BACKEND_BOOTSTRAP_API_KEY", default="")
        if bootstrap_key:
            tenant = env.str("BACKEND_BOOTSTRAP_TENANT_ID", default="")
            user = env.str("BACKEND_BOOTSTRAP_USER_ID", default="")
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
        json_keys = env.str("BACKEND_API_KEYS_JSON", default="")
        if json_keys:
            if keys:
                raise ValueError(
                    "Use either LEGAL_BACKEND_BOOTSTRAP_API_KEY or LEGAL_BACKEND_API_KEYS_JSON, not both."
                )
            keys = _parse_api_key_map(json_keys)

        if environment == ApiEnvironment.PROD and identity_source != "context_only":
            raise ValueError(
                "LEGAL_BACKEND_ENV=prod requires LEGAL_IDENTITY_SOURCE=context_only (or omit to default)."
            )
        if environment == ApiEnvironment.PROD and not env.bool("BACKEND_ALLOW_UNAUTHENTICATED", default=False):
            if not keys:
                raise ValueError(
                    "Production Legal backend requires API keys: set LEGAL_BACKEND_BOOTSTRAP_API_KEY "
                    "(+ tenant/user) or LEGAL_BACKEND_API_KEYS_JSON. "
                    "For local disaster debugging only, set LEGAL_BACKEND_ALLOW_UNAUTHENTICATED=true."
                )

        enable_rag = env.bool("ENABLE_RAG", default=False)
        enable_rag_ingest = env.bool("ENABLE_RAG_INGEST", default=False)
        enable_websearch = env.bool("ENABLE_WEBSEARCH", default=False)
        use_legal_tool_decision = env.bool("USE_TOOL_DECISION", default=False)
        tools_mode = env.str("TOOLS_MODE", default="off").lower() or "off"
        extra_tools = tuple(env.csv_set("ENABLED_TOOLS"))
        enable_modality = env.bool("ENABLE_MODALITY_TOOLS", default=False)
        enable_llm_guardrails = env.bool("ENABLE_LLM_GUARDRAILS", default=False)
        guardrail_primary = env.str("LLM_GUARDRAIL_PRIMARY", default="llm_guard")
        guardrail_semantic = env.str("LLM_GUARDRAIL_SEMANTIC", default="presidio")

        if profile == "research" and os.environ.get("LEGAL_ENABLE_RAG") is None:
            enable_rag = True
        if profile == "research" and os.environ.get("LEGAL_ENABLE_WEBSEARCH") is None:
            enable_websearch = True
        if profile == "research" and os.environ.get("LEGAL_USE_TOOL_DECISION") is None:
            use_legal_tool_decision = True

        return {
            "legal_product_profile": profile,
            "legal_llm_provider": llm,
            "legal_llm_model": llm_model,
            "legal_default_agent_id": agent_id,
            "identity_source": identity_source,
            "cors_allow_origins": cors,
            "allowed_hosts": hosts,
            "openapi_enabled_override": openapi_override,
            "session_sqlite_path": session_db,
            "api_keys_map": keys,
            "interaction_execute_default": env.bool("INTERACTION_EXECUTE_DEFAULT", default=True),
            "enable_rag": enable_rag,
            "enable_rag_ingest": enable_rag_ingest,
            "enable_websearch": enable_websearch,
            "use_legal_tool_decision": use_legal_tool_decision,
            "tools_mode": tools_mode,
            "extra_enabled_tool_ids": extra_tools,
            "enable_modality_tools": enable_modality,
            "enable_llm_guardrails": enable_llm_guardrails,
            "llm_guardrail_primary": guardrail_primary,
            "llm_guardrail_semantic": guardrail_semantic,
        }
