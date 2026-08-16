# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from dispute_sim_application.host.environment_profile import build_dispute_sim_environment_profile
from dispute_sim_application.host.factory import create_dispute_sim_backend_app
from dispute_sim_application.manifest import build_dispute_sim_manifest

load_dotenv()

_manifest = build_dispute_sim_manifest()
_env = _manifest.environment or build_dispute_sim_environment_profile()

app = create_dispute_sim_backend_app(
    registry_projection=bootstrap_production_registry_projection(
        application_id=_manifest.app_id,
        application_environment_id=_env.profile_id,
    ),
)


def run() -> None:
    import uvicorn

    host = os.environ.get("DISPUTE_SIM_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("DISPUTE_SIM_BACKEND_PORT", "8020"))
    uvicorn.run(
        "dispute_sim_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("DISPUTE_SIM_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
