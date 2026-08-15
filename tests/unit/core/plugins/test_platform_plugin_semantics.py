# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import importlib.metadata

import pytest
from packaging.version import Version

from intergrax.core.plugins.discovery import load_entry_point_plugins
from intergrax.core.plugins.errors import (
    InvalidPlatformVersionError,
    PlatformIncompatibilityError,
)
from intergrax.core.plugins.package_contract import (
    PlatformCompatibility,
    PluginPackageIdentity,
    build_platform_plugin_manifest,
)
from intergrax.core.plugins.platform_semantics import (
    PlatformCompatibilityReason,
    PlatformPluginConflictKind,
    PlatformPluginLifecycleState,
    check_manifest_platform_compatibility,
    check_platform_compatibility,
    normalize_platform_version,
    package_identities_conflict,
    require_platform_compatibility,
)

pytestmark = pytest.mark.unit


def test_lifecycle_state_canonical_values() -> None:
    assert tuple(PlatformPluginLifecycleState) == (
        "discovered",
        "validated",
        "enabled",
        "materialized",
        "active",
        "stopping",
        "stopped",
        "failed",
    )


def test_lifecycle_enum_excludes_qualification_states() -> None:
    values = {item.value for item in PlatformPluginLifecycleState}
    assert "qualified" not in values
    assert "production_qualified" not in values
    assert "live_qualified" not in values
    assert "installed" not in values


def test_conflict_kind_canonical_values() -> None:
    assert tuple(PlatformPluginConflictKind) == (
        "package_identity",
        "entry_point_name",
        "capability_identity",
        "domain_resource_id",
    )


def test_check_platform_compatibility_compatible_version() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1,<2")
    result = check_platform_compatibility(declared, "1.5")
    assert result.compatible is True
    assert result.reason is PlatformCompatibilityReason.COMPATIBLE
    assert result.declared_specifier == "<2,>=1"
    assert result.tested_platform_version == "1.5"


def test_check_platform_compatibility_incompatible_version() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1,<2")
    result = check_platform_compatibility(declared, "2.0")
    assert result.compatible is False
    assert result.reason is PlatformCompatibilityReason.INCOMPATIBLE_VERSION


def test_check_platform_compatibility_exact_pin() -> None:
    declared = PlatformCompatibility(intergrax_version="==1.2.3")
    assert check_platform_compatibility(declared, "1.2.3").compatible is True
    assert check_platform_compatibility(declared, "1.2.4").compatible is False


def test_check_platform_compatibility_normalizes_runtime_version() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1.0.0")
    result = check_platform_compatibility(declared, Version("1.0"))
    assert result.tested_platform_version == "1.0"
    assert result.compatible is True


def test_check_platform_compatibility_prerelease_semantics() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1.0,<2")
    excluded = check_platform_compatibility(declared, "1.5b1")
    assert excluded.compatible is False
    assert excluded.reason is PlatformCompatibilityReason.INCOMPATIBLE_VERSION

    prerelease_declared = PlatformCompatibility(intergrax_version=">=1.0b1,<2")
    included = check_platform_compatibility(prerelease_declared, "1.5b1")
    assert included.compatible is True


def test_check_platform_compatibility_invalid_platform_version() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1,<2")
    result = check_platform_compatibility(declared, "not-a-version")
    assert result.compatible is False
    assert result.reason is PlatformCompatibilityReason.INVALID_PLATFORM_VERSION


def test_normalize_platform_version_rejects_blank() -> None:
    with pytest.raises(InvalidPlatformVersionError):
        normalize_platform_version("   ")


def test_require_platform_compatibility_raises_on_mismatch() -> None:
    declared = PlatformCompatibility(intergrax_version=">=2")
    with pytest.raises(PlatformIncompatibilityError) as exc_info:
        require_platform_compatibility(declared, "1.0.0")
    assert exc_info.value.result is not None
    assert exc_info.value.result.compatible is False


def test_require_platform_compatibility_raises_on_invalid_version() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1")
    with pytest.raises(InvalidPlatformVersionError):
        require_platform_compatibility(declared, "bad")


def test_compatibility_result_is_immutable_and_auditable() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1")
    result = check_platform_compatibility(declared, "1.0.0")
    with pytest.raises(AttributeError):
        result.compatible = False  # type: ignore[misc]


def test_compatibility_check_does_not_mutate_metadata() -> None:
    declared = PlatformCompatibility(intergrax_version=">=1,<2")
    before = declared.model_dump()
    check_platform_compatibility(declared, "1.0.0")
    assert declared.model_dump() == before


def test_manifest_compatibility_check_is_package_level() -> None:
    manifest = build_platform_plugin_manifest(
        name="acme-intergrax",
        version="1.0.0",
        intergrax_version=">=1,<2",
    )
    result = check_manifest_platform_compatibility(manifest, "1.5")
    assert result.compatible is True


def test_package_identities_conflict_same_name_different_version() -> None:
    left = PluginPackageIdentity(name="acme-intergrax", version="1.0.0")
    right = PluginPackageIdentity(name="acme-intergrax", version="2.0.0")
    assert package_identities_conflict(left, right) is True


def test_package_identities_no_conflict_for_different_names() -> None:
    left = PluginPackageIdentity(name="acme-intergrax", version="1.0.0")
    right = PluginPackageIdentity(name="other-plugin", version="1.0.0")
    assert package_identities_conflict(left, right) is False


class _EntryPoint:
    def __init__(self, name: str, value: str, group: str) -> None:
        self.name = name
        self.value = value
        self.group = group


class _EntryPoints:
    def __init__(self, entries: list[_EntryPoint]) -> None:
        self._entries = entries

    def select(self, *, group: str) -> list[_EntryPoint]:
        return [entry for entry in self._entries if entry.group == group]


class _DiscoveredPlugin:
    pass


def test_duplicate_entry_point_conflict_kind_without_behavior_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = _EntryPoints(
        [
            _EntryPoint("duplicate", f"{__name__}:_DiscoveredPlugin", "intergrax.tools"),
            _EntryPoint("duplicate", f"{__name__}:_DiscoveredPlugin", "intergrax.tools"),
        ]
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: entries)

    with pytest.raises(Exception) as exc_info:
        load_entry_point_plugins("intergrax.tools")
    exc = exc_info.value
    assert exc.conflict_kind is PlatformPluginConflictKind.ENTRY_POINT_NAME
    assert exc.plugin_name == "duplicate"


def test_lifecycle_enum_has_no_global_state_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("lifecycle vocabulary must not trigger discovery")

    monkeypatch.setattr(
        "intergrax.core.plugins.discovery.load_entry_point_plugins",
        _fail_if_called,
    )
    assert PlatformPluginLifecycleState.DISCOVERED.value == "discovered"
