# © Artur Czarnecki. All rights reserved.

"""LKW Kafka background worker process entrypoint (LKW.4E)."""

from __future__ import annotations

import logging
import sys

from local_workspace_application.host.background_worker_factory import (
    build_local_workspace_background_worker_wiring,
)
from local_workspace_application.host.message_bus_wiring import local_workspace_message_bus_enabled
from local_workspace_application.manifest import LOCAL_WORKSPACE_APPLICATION_MANIFEST

logger = logging.getLogger(__name__)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not local_workspace_message_bus_enabled():
        logger.error("LOCAL_WORKSPACE_ENABLE_MESSAGE_BUS must be true for the background worker")
        return 1

    wiring = build_local_workspace_background_worker_wiring(
        manifest=LOCAL_WORKSPACE_APPLICATION_MANIFEST,
    )
    logger.info("Starting LKW Kafka background worker for lkw.background_ingest.v1")
    wiring.worker.start()
    return 0


if __name__ == "__main__":
    sys.exit(main())
