# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness-safe RuntimeConfig defaults for Tier-3 lab/scaffold (Phase Q-N.10)."""

from __future__ import annotations


def harness_production_mode() -> bool:
    """Lab and scaffold agents must run with production governance relaxed."""
    return False
