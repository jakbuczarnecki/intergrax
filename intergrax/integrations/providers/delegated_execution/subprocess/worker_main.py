# © Artur Czarnecki. All rights reserved.

"""Subprocess worker entrypoint (separate Python interpreter boundary)."""

from __future__ import annotations

import os

from intergrax.integrations.providers.delegated_execution.subprocess.worker_server import (
    run_worker_server,
)


def main() -> None:
    max_concurrent = int(os.environ.get("SUBPROCESS_DELEGATED_MAX_CONCURRENT", "8"))
    run_worker_server(max_concurrent=max_concurrent)


if __name__ == "__main__":
    main()
