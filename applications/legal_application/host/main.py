# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ASGI entrypoint for the Legal backend host."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from legal_application.host.factory import create_legal_backend_app
from legal_application.host.settings import LegalBackendSettings
from legal_application.host.wiring import build_legal_environment_profile, build_legal_manifest

# Load `.env` when present (does not override existing process env).
load_dotenv()

_settings = LegalBackendSettings.from_env()
_manifest = build_legal_manifest(_settings)
_env = _manifest.environment or build_legal_environment_profile(_settings)
_agent_platform_runtime = build_production_agent_platform_runtime()

app = create_legal_backend_app(
    registry_projection=bootstrap_production_registry_projection(
        application_id=_manifest.app_id,
        application_environment_id=_env.profile_id,
        stores=_agent_platform_runtime.stores,
    ),
    settings=_settings,
)


def run() -> None:
    """CLI entry using uvicorn when installed."""
    import uvicorn

    host = os.environ.get("LEGAL_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("LEGAL_BACKEND_PORT", "8000"))
    reload = os.environ.get("LEGAL_BACKEND_RELOAD", "").strip().lower() in {"1", "true", "yes"}
    uvicorn.run(
        "legal_application.host.main:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run()
