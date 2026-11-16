# © Artur Czarnecki. All rights reserved.

"""Repo-root launcher for governed-contractor CLI commands."""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_tier_paths() -> None:
    candidates = [
        Path.cwd(),
        Path(__file__).resolve().parents[3],
    ]
    for base in candidates:
        apps = base / "applications"
        agents = base / "agents"
        if apps.is_dir() and agents.is_dir():
            for rel in (".", "applications", "agents"):
                path = str((base / rel).resolve())
                if path not in sys.path:
                    sys.path.insert(0, path)
            return


def main(argv: list[str] | None = None) -> int:
    _ensure_tier_paths()
    from intergrax.cli.main import main as intergrax_main

    from applications.governed_contractor_application.host.cli_commands import (
        GOVERNED_CONTRACTOR_COMMANDS,
    )

    return intergrax_main(argv, extra_command_sets=(GOVERNED_CONTRACTOR_COMMANDS,))


if __name__ == "__main__":
    raise SystemExit(main())
