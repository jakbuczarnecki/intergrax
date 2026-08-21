# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import os

import uvicorn

from proof_infrastructure.controlled_governance_approval_service.app import create_app
from proof_infrastructure.controlled_governance_approval_service.mongo_state import (
    create_governance_approval_store_from_env,
)
from proof_infrastructure.controlled_governance_approval_service.seed import (
    ORION_FIXTURE_SUBJECT_ID,
    seed_orion_governance_fixture,
)


def main() -> None:
    store = create_governance_approval_store_from_env()
    if store.get_governance(ORION_FIXTURE_SUBJECT_ID) is None:
        seed_orion_governance_fixture(store)
    app = create_app(store=store)
    host = os.environ.get("GOVERNANCE_APPROVAL_HOST", "0.0.0.0")
    port = int(os.environ.get("GOVERNANCE_APPROVAL_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
