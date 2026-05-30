# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register default tool catalog bundles (Phase O.2+)."""

from __future__ import annotations

_BOOTSTRAPPED = False


def register_default_tools(*, override: bool = False) -> None:
    """
    Idempotent registration of shipped tool catalog bundles.

    Call from Tier-3 application factories before ``build_registry_from_profile()``.

    Provider bundles (``jira``, ``rag``, ``websearch``, …) are added in Phase O.3–O.4.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override:
        return

    from intergrax.tools.providers.rag.register import register_rag_tool_bundle

    register_rag_tool_bundle(override=override)
    _BOOTSTRAPPED = True


def reset_default_tools_bootstrap() -> None:
    """Test helper — allow ``register_default_tools()`` to run again."""
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False
