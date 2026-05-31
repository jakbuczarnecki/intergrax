# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Register default tool catalog bundles (Phase O.2+)."""

from __future__ import annotations

_BOOTSTRAPPED = False


def register_default_tools(*, override: bool = False) -> None:
    """
    Idempotent registration of shipped tool catalog bundles.

    Call from Tier-3 application factories before ``build_registry_from_profile()``.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED and not override:
        return

    from intergrax.tools.providers.braintrust.register import register_braintrust_tool_bundle
    from intergrax.tools.providers.gitlab.register import register_gitlab_tool_bundle
    from intergrax.tools.providers.pagerduty.register import register_pagerduty_tool_bundle
    from intergrax.tools.providers.confluence.register import register_confluence_tool_bundle
    from intergrax.tools.providers.jira.register import register_jira_tool_bundle
    from intergrax.tools.providers.notify.register import register_notify_tool_bundle
    from intergrax.tools.providers.observability.register import register_observability_tool_bundle
    from intergrax.tools.providers.rag.register import register_rag_tool_bundle
    from intergrax.tools.providers.sandbox.register import register_sandbox_tool_bundle
    from intergrax.tools.providers.websearch.register import register_websearch_tool_bundle

    for register_fn in (
        register_rag_tool_bundle,
        register_websearch_tool_bundle,
        register_jira_tool_bundle,
        register_gitlab_tool_bundle,
        register_confluence_tool_bundle,
        register_notify_tool_bundle,
        register_pagerduty_tool_bundle,
        register_observability_tool_bundle,
        register_braintrust_tool_bundle,
        register_sandbox_tool_bundle,
    ):
        register_fn(override=override)

    _BOOTSTRAPPED = True


def reset_default_tools_bootstrap() -> None:
    """Test helper — allow ``register_default_tools()`` to run again."""
    global _BOOTSTRAPPED
    _BOOTSTRAPPED = False
