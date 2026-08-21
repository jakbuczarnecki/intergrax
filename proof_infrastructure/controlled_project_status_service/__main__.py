# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import uvicorn

from proof_infrastructure.controlled_project_status_service.app import create_app
from proof_infrastructure.controlled_project_status_service.mongo_state import (
    create_project_status_store_from_env,
)
from proof_infrastructure.controlled_project_status_service.seed import (
    ORION_FIXTURE_PROJECT_ID,
    seed_orion_fixture,
)


def main() -> None:
    store = create_project_status_store_from_env()
    if store.get_status(ORION_FIXTURE_PROJECT_ID) is None:
        seed_orion_fixture(store)
    app = create_app(store=store)
    host = os.environ.get("PROJECT_STATUS_HOST", "0.0.0.0")
    port = int(os.environ.get("PROJECT_STATUS_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
