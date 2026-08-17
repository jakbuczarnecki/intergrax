# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from intergrax.applications._shared.production_agent_platform_runtime import (
    build_production_agent_platform_runtime,
)
from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.factory import create_governed_contractor_backend_app
from governed_contractor_application.manifest import build_governed_contractor_manifest

load_dotenv()

_manifest = build_governed_contractor_manifest()
_env = _manifest.environment or build_governed_contractor_environment_profile()
_agent_platform_runtime = build_production_agent_platform_runtime()

app = create_governed_contractor_backend_app(
    registry_projection=bootstrap_production_registry_projection(
        application_id=_manifest.app_id,
        application_environment_id=_env.profile_id,
        stores=_agent_platform_runtime.stores,
    ),
)


def run() -> None:
    import uvicorn

    host = os.environ.get("GOVERNED_CONTRACTOR_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("GOVERNED_CONTRACTOR_BACKEND_PORT", "8000"))
    uvicorn.run(
        "governed_contractor_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("GOVERNED_CONTRACTOR_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
