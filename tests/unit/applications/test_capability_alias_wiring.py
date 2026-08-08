# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-3 — capability alias registry and sunset routing."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.applications.contracts.capability_alias import (
    CapabilityAlias,
    CapabilityGovernanceProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import ApplicationManifest
from intergrax.applications._shared.capability_alias_wiring import (
    build_capability_alias_registry,
    check_environment_capability_aliases,
    resolve_capability_alias,
    validate_capability_governance_profile,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
_WINDOW_START = "2026-06-01T00:00:00Z"
_WINDOW_END = "2026-07-01T00:00:00Z"


def _alias_profile() -> CapabilityGovernanceProfile:
    return CapabilityGovernanceProfile(
        minimum_alias_window_days=14,
        aliases=[
            CapabilityAlias(
                alias="research.pipeline",
                canonical="research.orchestrate",
                effective_from=_WINDOW_START,
                sunset_at=_WINDOW_END,
                notice_ref="docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md",
            ),
        ],
    )


def test_resolve_redirects_legacy_capability_during_window() -> None:
    registry = build_capability_alias_registry(_alias_profile())
    resolution = resolve_capability_alias(
        "research.pipeline",
        registry,
        now=_NOW,
        strict=True,
    )
    assert resolution.redirected is True
    assert resolution.resolved == "research.orchestrate"
    assert resolution.blocked is False


def test_resolve_blocks_after_sunset_in_strict_mode() -> None:
    registry = build_capability_alias_registry(_alias_profile())
    resolution = resolve_capability_alias(
        "research.pipeline",
        registry,
        now=datetime(2026, 7, 2, tzinfo=UTC),
        strict=True,
    )
    assert resolution.blocked is True
    assert resolution.redirected is False


def test_resolve_allows_passthrough_after_sunset_in_balanced_mode() -> None:
    registry = build_capability_alias_registry(_alias_profile())
    resolution = resolve_capability_alias(
        "research.pipeline",
        registry,
        now=datetime(2026, 7, 2, tzinfo=UTC),
        strict=False,
    )
    assert resolution.blocked is False
    assert resolution.resolved == "research.pipeline"


def test_validate_alias_window_minimum() -> None:
    profile = CapabilityGovernanceProfile(
        minimum_alias_window_days=14,
        aliases=[
            CapabilityAlias(
                alias="legacy.cap",
                canonical="modern.cap",
                effective_from="2026-06-10T00:00:00Z",
                sunset_at="2026-06-15T00:00:00Z",
            ),
        ],
    )
    violations = validate_capability_governance_profile(profile)
    assert any("alias window shorter" in item for item in violations)


def test_manifest_must_not_declare_alias_capabilities() -> None:
    profile = _alias_profile()
    manifest = ApplicationManifest.model_validate(
        {
            "app_id": "demo",
            "name": "Demo",
            "version": "0.1.0",
            "route_prefix": "/v1/demo",
            "env_prefix": "DEMO_",
            "agents": [
                {
                    "import_path": "agents.echo.echo_agent.EchoAgent",
                    "capabilities": ["research.pipeline"],
                    "enabled": True,
                },
            ],
        },
    )
    violations = check_environment_capability_aliases("demo_application", manifest, profile)
    assert any("lists alias" in item for item in violations)


def test_before_effective_from_does_not_redirect() -> None:
    registry = build_capability_alias_registry(_alias_profile())
    resolution = resolve_capability_alias(
        "research.pipeline",
        registry,
        now=datetime(2026, 5, 15, tzinfo=UTC),
        strict=True,
    )
    assert resolution.redirected is False
    assert resolution.resolved == "research.pipeline"
