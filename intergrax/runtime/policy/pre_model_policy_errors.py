# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""PRE_MODEL configuration failures (GR-10-R2 / GR-10-R4-R1 import boundary)."""

from __future__ import annotations


class PreModelPolicyConfigurationError(RuntimeError):
    """PRE_MODEL cannot run — missing policy dependency or governance identity."""
