# © Artur Czarnecki. All rights reserved.

"""Test helpers for IntegrationProfile binding assertions."""

from __future__ import annotations

from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.registry.profile import IntegrationProfile


def assert_profile_slug(profile: IntegrationProfile, field_name: str, expected: str) -> None:
    binding: IntegrationBinding | None = profile.binding_for_field(field_name)
    assert binding is not None, f"expected {field_name!r} to be set"
    assert binding.resolved_slug() == expected
