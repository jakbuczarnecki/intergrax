# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Microsoft 365 Graph integration configuration (Phase M.6)."""

from __future__ import annotations

import os

from intergrax.integrations._shared.config import BaseIntegrationConfig

ENV_MS365_TENANT_ID = "INTERGRAX_MS365_TENANT_ID"
ENV_MS365_CLIENT_ID = "INTERGRAX_MS365_CLIENT_ID"
ENV_MS365_CLIENT_SECRET = "INTERGRAX_MS365_CLIENT_SECRET"
ENV_MS365_DEFAULT_USER = "INTERGRAX_MS365_DEFAULT_USER"

DEFAULT_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
DEFAULT_TIMEOUT_SECONDS = 30.0


class Ms365GraphIntegrationConfig(BaseIntegrationConfig):
    """
    Microsoft Graph app-only (client credentials) settings.

    ``default_user`` is optional — callers may pass ``user_id`` explicitly.
    """

    tenant_id: str = ""
    client_id: str = ""
    client_secret: str = ""
    default_user: str = ""
    graph_base_url: str = DEFAULT_GRAPH_BASE_URL

    @property
    def token_url(self) -> str:
        return f"https://login.microsoftonline.com/{self.tenant_id.strip()}/oauth2/v2.0/token"

    @classmethod
    def from_env(cls, **overrides: object) -> Ms365GraphIntegrationConfig:
        tenant_id = os.environ.get(ENV_MS365_TENANT_ID, "").strip()
        client_id = os.environ.get(ENV_MS365_CLIENT_ID, "").strip()
        client_secret = os.environ.get(ENV_MS365_CLIENT_SECRET, "").strip()
        default_user = os.environ.get(ENV_MS365_DEFAULT_USER, "").strip()
        timeout_raw = os.environ.get("INTERGRAX_MS365_TIMEOUT_SECONDS", "").strip()
        payload: dict[str, object] = {
            "tenant_id": tenant_id,
            "client_id": client_id,
            "client_secret": client_secret,
            "default_user": default_user,
        }
        if timeout_raw:
            payload["timeout_seconds"] = float(timeout_raw)
        else:
            payload["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
        payload.update(overrides)
        return cls.model_validate(payload)
