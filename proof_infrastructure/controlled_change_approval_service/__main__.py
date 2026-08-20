# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import uvicorn

from proof_infrastructure.controlled_change_approval_service.app import create_app
from proof_infrastructure.controlled_change_approval_service.mongo_state import (
    create_change_approval_store_from_env,
)
from proof_infrastructure.controlled_change_approval_service.seed import (
    ORION_FIXTURE_CHANGE_ID,
    seed_orion_change_fixture,
)


def main() -> None:
    store = create_change_approval_store_from_env()
    if store.get_change(ORION_FIXTURE_CHANGE_ID) is None:
        seed_orion_change_fixture(store)
    app = create_app(store=store)
    host = os.environ.get("CHANGE_APPROVAL_HOST", "0.0.0.0")
    port = int(os.environ.get("CHANGE_APPROVAL_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
