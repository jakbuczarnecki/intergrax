#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-25.1 — golden scenario library gate."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    from intergrax.runtime.architecture.evaluation_scenario_loader import load_scenario_library

    library = load_scenario_library(REPO_ROOT)
    if len(library.scenarios) < 2:
        print("scenario library too small", file=sys.stderr)
        return 1
    print(f"OK: scenario library {library.library_id} v{library.version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
