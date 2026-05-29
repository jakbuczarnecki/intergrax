# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register default P0 integration providers (Phase M.4+)."""

from __future__ import annotations

from intergrax.integrations.providers.redis.register import register_redis_integration

_BOOTSTRAPPED = False


def register_default_integrations(*, override: bool = False) -> None:
    """
    Idempotent registration of shipped integration providers.

    Call from Tier-3 application factories before ``resolve()``.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override:
        return
    register_redis_integration(override=override)
    _BOOTSTRAPPED = True


def reset_default_integrations_state() -> None:
    """Test helper — allow re-bootstrap after ``clear_catalog()``."""
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False
