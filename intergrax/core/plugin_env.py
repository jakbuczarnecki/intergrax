# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Environment flags for Tier-0 plugin catalog discovery."""

from __future__ import annotations

import os

INTERGRAX_DISCOVER_PLUGINS_ENV = "INTERGRAX_DISCOVER_PLUGINS"


def discover_plugins_enabled() -> bool:
    """True when ``INTERGRAX_DISCOVER_PLUGINS`` is set to a truthy value."""
    value = os.environ.get(INTERGRAX_DISCOVER_PLUGINS_ENV, "").strip().lower()
    return value in ("1", "true", "yes", "on")
