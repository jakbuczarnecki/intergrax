# © Artur Czarnecki. All rights reserved.

"""CLI entrypoint: python -m local_workspace_application.file_watcher (LKW.7B2B)."""

from __future__ import annotations

import json
import logging

from local_workspace_application.file_watcher.sidecar import (
    run_local_workspace_file_watcher_sidecar,
)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_local_workspace_file_watcher_sidecar()
    print(
        json.dumps(
            result.model_dump(mode="json"),
            sort_keys=True,
        ),
        flush=True,
    )
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
