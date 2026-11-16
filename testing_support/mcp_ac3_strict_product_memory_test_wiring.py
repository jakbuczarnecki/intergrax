# © Artur Czarnecki. All rights reserved.

"""Test-only strict PRODUCT memory wiring with runner-backed qualification evidence.

Uses ``MemoryProviderQualificationRunner`` evidence (same helpers as MEM 5A admission
tests) — not synthetic bypass of admission.
"""

from __future__ import annotations

from typing import Any

from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.memory.contracts.provider_identity import BUILTIN_SQLITE_USER_PROFILE_ID


def resolve_memory_platform_wiring_for_strict_product_tests(
    env: ApplicationEnvironmentProfile,
    *,
    integration_profile: Any = None,
    _original_resolve=None,
    **kwargs: Any,
):
    from tests.unit.applications.test_mem_audit5a_production_provider_admission import (
        _durability_for_provider,
        _evidence_for_provider,
    )

    if _original_resolve is None:
        raise RuntimeError("strict_product_memory_test_wiring_missing_original_resolver")

    if env.application_profile.value == "product" and env.execution_mode.value == "strict":
        kwargs.setdefault(
            "qualification_evidence_registry",
            _evidence_for_provider(BUILTIN_SQLITE_USER_PROFILE_ID),
        )
        kwargs.setdefault(
            "durability_evidence_registry",
            _durability_for_provider(BUILTIN_SQLITE_USER_PROFILE_ID),
        )
    return _original_resolve(
        env,
        integration_profile=integration_profile,
        **kwargs,
    )


def apply_strict_product_memory_test_patches(monkeypatch) -> None:
    """Patch memory resolution sites used by harness host bootstrap in unit tests."""
    import intergrax.applications._shared.environment_wiring as environment_wiring
    import intergrax.applications._shared.memory_wiring as memory_wiring
    import intergrax.applications._shared.runtime_config_bridge as runtime_config_bridge

    original = memory_wiring.resolve_memory_platform_wiring

    def _wrapped(env, *, integration_profile=None, **kwargs):
        return resolve_memory_platform_wiring_for_strict_product_tests(
            env,
            integration_profile=integration_profile,
            _original_resolve=original,
            **kwargs,
        )

    monkeypatch.setattr(memory_wiring, "resolve_memory_platform_wiring", _wrapped)
    monkeypatch.setattr(environment_wiring, "resolve_memory_platform_wiring", _wrapped)
    monkeypatch.setattr(runtime_config_bridge, "resolve_memory_platform_wiring", _wrapped)
