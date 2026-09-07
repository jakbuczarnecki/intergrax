# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Reusable CodeCraft execution-environment fixtures for unit tests."""

from __future__ import annotations

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import IsolationBundle
from intergrax.applications.contracts.environment_profile.sub_profiles import SandboxProfile


def codecraft_sandbox_execution_profile(
    *,
    profile_id: str = "codecraft-test",
    enable_exec_tool: bool = True,
) -> ApplicationEnvironmentProfile:
    """Minimum typed ``ApplicationEnvironmentProfile`` for sandbox exec authority."""
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id=profile_id)
    return profile.model_copy(
        update={
            "isolation": IsolationBundle(
                sandbox=SandboxProfile(enable_exec_tool=enable_exec_tool),
            ),
        },
    )
