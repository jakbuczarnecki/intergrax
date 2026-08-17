# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

from dotenv import load_dotenv

from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from research_application.host.factory import create_research_backend_app
from research_application.host.wiring import build_research_environment_profile
from research_application.manifest import RESEARCH_APPLICATION_MANIFEST

load_dotenv()

_manifest = RESEARCH_APPLICATION_MANIFEST
_env = _manifest.environment or build_research_environment_profile()

app = create_research_backend_app(
    registry_projection=bootstrap_production_registry_projection(
        application_id=_manifest.app_id,
        application_environment_id=_env.profile_id,
    ),
)


def run() -> None:
    import uvicorn

    host = os.environ.get("RESEARCH_BACKEND_HOST", "0.0.0.0")
    port = int(os.environ.get("RESEARCH_BACKEND_PORT", "8010"))
    uvicorn.run(
        "research_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("RESEARCH_BACKEND_RELOAD", "").lower() in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
