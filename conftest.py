# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure(config) -> None:
    """Put product and agent roots on ``sys.path`` for Tier-2 imports.

    - ``applications/`` — execution environments (``legal_application``, ``legal_agent`` shim)
    - ``agents/`` — reusable capability modules (``legal``, ``echo``)
    """
    root = Path(__file__).resolve().parent
    (root / "build").mkdir(parents=True, exist_ok=True)

    for subdir in ("applications", "agents"):
        path_root = root / subdir
        if not path_root.is_dir():
            continue
        path = str(path_root.resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    # ``legal_agent`` shim loads lazily via PEP 562 when imported.
