# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import sys
from pathlib import Path


def pytest_configure(config) -> None:
    """Put ``applications/`` on ``sys.path`` so each product is ``applications/<import_name>/``.

    Example: ``applications/legal_agent/`` is the ``legal_agent`` package; ``import legal_agent`` works.
    """
    root = Path(__file__).resolve().parent
    apps = root / "applications"
    if not apps.is_dir():
        return
    path = str(apps.resolve())
    if path not in sys.path:
        sys.path.insert(0, path)
