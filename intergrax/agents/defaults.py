# © Artur Czarnecki. All rights reserved.

"""Tier-2 agent runtime defaults (Phase DX-6.1) — no Tier-3 imports."""

from __future__ import annotations


def harness_production_mode() -> bool:
    """Lab and scaffold agents run with relaxed production governance."""
    return False
