# © Artur Czarnecki. All rights reserved.

"""Test-only helper: reset the process-global integration, tool and skill catalogs.

Used by autouse fixtures that must start and end each test from an empty catalog state, so
a failed or partial bootstrap in one test cannot leak a half-filled catalog into the next.
Only public ``*_for_tests`` / ``clear_*`` / ``reset_*`` helpers are called.
"""

from __future__ import annotations

from intergrax.core.catalog_bootstrap import reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.plugins.discovery import reset_entry_point_spec_cache_for_tests
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.skills.registry.bootstrap import reset_default_skills_for_tests
from intergrax.skills.registry.catalog import clear_skill_catalog
from intergrax.tools.registry.bootstrap import reset_default_tools_bootstrap
from intergrax.tools.registry.catalog import clear_tool_catalog


def reset_integration_tool_skill_catalogs() -> None:
    clear_catalog()
    clear_tool_catalog()
    clear_skill_catalog()
    reset_default_integrations_state()
    reset_default_tools_bootstrap()
    reset_default_skills_for_tests()
    reset_tier0_catalog_bootstrap_for_tests()
    reset_entry_point_spec_cache_for_tests()
