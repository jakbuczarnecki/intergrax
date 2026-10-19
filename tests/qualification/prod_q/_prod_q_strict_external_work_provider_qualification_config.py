# © Artur Czarnecki. All rights reserved.

"""Typed strict external-work provider pin for PROD-Q qualification (laboratory only).

Must match ``DeterministicExternalWorkFake.discover()`` — not production defaults.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

from external_contractor_adapter.tests.fakes.deterministic_external_work import (
    DeterministicExternalWorkFake,
)
from governed_contractor_application.host.resolve_strict_external_work_provider_capabilities import (
    StrictExternalWorkProviderResolutionError,
    resolve_strict_external_work_provider_capabilities,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from intergrax.integrations.contracts.external_work import ExternalWorkIntegration

QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_ID: Final = "gec3_deterministic_fake"
QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST: Final = (
    "sha256:" + ("cd" * 32)
)


def with_qualified_strict_external_work_provider_pin(
    settings: GovernedContractorBackendSettings,
) -> GovernedContractorBackendSettings:
    """Apply the sanctioned laboratory provider pin (explicit; never from ambient env)."""
    return replace(
        settings,
        strict_external_work_provider_id=QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_ID,
        strict_external_work_provider_descriptor_digest=(
            QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST
        ),
    )


def assert_qualified_pin_resolves_against_integration(
    integration: ExternalWorkIntegration,
    *,
    settings: GovernedContractorBackendSettings | None = None,
) -> None:
    """Fail fast when qualification constants drift from the wired integration."""
    resolved_settings = with_qualified_strict_external_work_provider_pin(
        settings or GovernedContractorBackendSettings(),
    )
    resolve_strict_external_work_provider_capabilities(
        integration,
        settings=resolved_settings,
    )


def assert_unpinned_strict_external_work_settings_fail_closed(
    integration: ExternalWorkIntegration,
) -> None:
    """Guard against silent global defaults or empty laboratory settings."""
    unpinned = GovernedContractorBackendSettings()
    try:
        resolve_strict_external_work_provider_capabilities(
            integration,
            settings=unpinned,
        )
    except StrictExternalWorkProviderResolutionError:
        return
    raise AssertionError(
        "unpinned strict external-work settings must fail closed; "
        "do not add implicit provider defaults",
    )


def default_qualification_external_work_integration() -> DeterministicExternalWorkFake:
    """Single sanctioned in-memory integration for strict HTTP qualification."""
    return DeterministicExternalWorkFake()


__all__ = [
    "QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_DESCRIPTOR_DIGEST",
    "QUALIFIED_STRICT_EXTERNAL_WORK_PROVIDER_ID",
    "assert_qualified_pin_resolves_against_integration",
    "assert_unpinned_strict_external_work_settings_fail_closed",
    "default_qualification_external_work_integration",
    "with_qualified_strict_external_work_provider_pin",
]
