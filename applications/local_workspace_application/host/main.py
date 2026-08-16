# © Artur Czarnecki. All rights reserved.

import os

from dotenv import load_dotenv

load_dotenv()

from intergrax.applications._shared.production_host_composition import (
    bootstrap_production_registry_projection,
)
from local_workspace_application.host.environment_profile import (
    build_local_workspace_environment_profile,
)
from local_workspace_application.host.factory import create_local_workspace_backend_app
from local_workspace_application.host.settings import LocalWorkspaceBackendSettings
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

_settings = LocalWorkspaceBackendSettings.from_env()
_env = build_local_workspace_environment_profile(_settings)

app = create_local_workspace_backend_app(
    registry_projection=bootstrap_production_registry_projection(
        application_id=LOCAL_WORKSPACE_APPLICATION_MANIFEST.app_id,
        application_environment_id=_env.profile_id,
    ),
    settings=_settings,
)


def run() -> None:
    import uvicorn

    host = os.environ.get("LOCAL_WORKSPACE_BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("LOCAL_WORKSPACE_BACKEND_PORT", "8020"))
    uvicorn.run(
        "local_workspace_application.host.main:app",
        host=host,
        port=port,
        reload=os.environ.get("LOCAL_WORKSPACE_BACKEND_RELOAD", "").lower()
        in {"1", "true", "yes"},
    )


if __name__ == "__main__":
    run()
