# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
from intergrax.core.plugins.manifest_io import (
    parse_platform_plugin_manifest_data,
    parse_platform_plugin_pyproject,
    parse_platform_plugin_pyproject_toml,
)
from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility
from intergrax.core.plugins.package_contract import (
    CapabilityDescriptor,
    build_platform_plugin_manifest,
)

pytestmark = pytest.mark.unit


def test_plugin_package_identity_normalizes_name_and_version() -> None:
    identity = DistributionPackageIdentity(name="Acme-Intergrax", version="1.0.0")
    assert identity.name == "acme-intergrax"
    assert identity.version == "1.0.0"


@pytest.mark.parametrize("name", ["", "   ", "!!!"])
def test_plugin_package_identity_rejects_invalid_name(name: str) -> None:
    with pytest.raises(ValidationError):
        DistributionPackageIdentity(name=name, version="1.0.0")


@pytest.mark.parametrize("version", ["", "   ", "not-a-version"])
def test_plugin_package_identity_rejects_invalid_version(version: str) -> None:
    with pytest.raises(ValidationError):
        DistributionPackageIdentity(name="acme-intergrax", version=version)


def test_platform_compatibility_accepts_valid_specifier() -> None:
    compatibility = PlatformCompatibility(intergrax_version=">=1.0,<2")
    assert compatibility.intergrax_version == "<2,>=1.0"
    assert str(compatibility.declared_specifier) == "<2,>=1.0"


@pytest.mark.parametrize("specifier", ["", "   ", "not-a-specifier"])
def test_platform_compatibility_rejects_invalid_specifier(specifier: str) -> None:
    with pytest.raises(ValidationError):
        PlatformCompatibility(intergrax_version=specifier)


def test_capability_descriptor_valid() -> None:
    descriptor = CapabilityDescriptor(
        domain="integrations",
        entry_point_group="intergrax.integrations",
        entry_point_name="acme_foo",
        capability_ids=["acme_foo"],
    )
    assert descriptor.capability_ids == ("acme_foo",)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("domain", ""),
        ("entry_point_group", "   "),
        ("entry_point_name", ""),
    ],
)
def test_capability_descriptor_rejects_blank_fields(field_name: str, value: str) -> None:
    payload = {
        "domain": "integrations",
        "entry_point_group": "intergrax.integrations",
        "entry_point_name": "acme_foo",
        field_name: value,
    }
    with pytest.raises(ValidationError):
        CapabilityDescriptor.model_validate(payload)


def test_multi_capability_package_accepts_multiple_domains() -> None:
    manifest = build_platform_plugin_manifest(
        name="acme-intergrax",
        version="1.0.0",
        intergrax_version=">=1.0,<2",
        capabilities=[
            CapabilityDescriptor(
                domain="integrations",
                entry_point_group="intergrax.integrations",
                entry_point_name="acme_foo",
                capability_ids=["acme_foo"],
            ),
            CapabilityDescriptor(
                domain="tools",
                entry_point_group="intergrax.tools",
                entry_point_name="acme_tool",
            ),
            CapabilityDescriptor(
                domain="skills",
                entry_point_group="intergrax.skills",
                entry_point_name="acme_skill",
            ),
        ],
    )
    assert len(manifest.capabilities) == 3
    assert {descriptor.domain for descriptor in manifest.capabilities} == {
        "integrations",
        "tools",
        "skills",
    }


def test_duplicate_capability_descriptors_rejected() -> None:
    descriptor = CapabilityDescriptor(
        domain="tools",
        entry_point_group="intergrax.tools",
        entry_point_name="acme_tool",
    )
    with pytest.raises(PlatformPluginManifestValidationError, match="duplicate capability descriptor"):
        build_platform_plugin_manifest(
            name="acme-intergrax",
            version="1.0.0",
            intergrax_version=">=1.0,<2",
            capabilities=[descriptor, descriptor],
        )


def test_minimal_manifest_from_mapping() -> None:
    manifest = parse_platform_plugin_manifest_data(
        {
            "name": "acme-intergrax",
            "version": "1.0.0",
            "intergrax_version": ">=1.0,<2",
        }
    )
    assert manifest.package.name == "acme-intergrax"
    assert manifest.capabilities == ()


def test_multi_capability_manifest_with_optional_metadata() -> None:
    manifest = parse_platform_plugin_manifest_data(
        {
            "name": "acme-intergrax",
            "version": "1.0.0",
            "intergrax_version": ">=1.0,<2",
            "author": "Acme Corp",
            "documentation_uri": "https://docs.example.com/acme-intergrax",
            "labels": ["community", "preview"],
            "capabilities": [
                {
                    "domain": "integrations",
                    "entry_point_group": "intergrax.integrations",
                    "entry_point_name": "acme_foo",
                    "capability_ids": ["acme_foo"],
                },
                {
                    "domain": "tools",
                    "entry_point_group": "intergrax.tools",
                    "entry_point_name": "acme_tool",
                },
            ],
        }
    )
    assert manifest.author == "Acme Corp"
    assert manifest.labels == ("community", "preview")
    assert len(manifest.capabilities) == 2


def test_unknown_manifest_fields_fail_closed() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="unknown manifest field"):
        parse_platform_plugin_manifest_data(
            {
                "name": "acme-intergrax",
                "version": "1.0.0",
                "intergrax_version": ">=1.0,<2",
                "typo_field": "value",
            }
        )


def test_secret_like_fields_rejected() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="secret-like manifest field"):
        parse_platform_plugin_manifest_data({"client_secret": "value"})


@pytest.mark.parametrize("key", ["api_key", "password", "token", "api-key"])
def test_secret_like_keys_rejected(key: str) -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="secret-like manifest field"):
        parse_platform_plugin_manifest_data({key: "value"})


def test_nested_credential_like_key_rejected() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="options.api_key"):
        parse_platform_plugin_manifest_data({"options": {"api_key": "x"}})


def test_safe_non_secret_manifest_accepted() -> None:
    manifest = parse_platform_plugin_manifest_data(
        {
            "name": "acme-intergrax",
            "version": "1.0.0",
            "intergrax_version": ">=1.0,<2",
            "author": "Acme",
            "labels": ["community"],
        }
    )
    assert manifest.package.name == "acme-intergrax"
    assert manifest.labels == ("community",)


def test_plugin_does_not_scan_secret_looking_scalar_values() -> None:
    manifest = parse_platform_plugin_manifest_data(
        {
            "name": "acme-intergrax",
            "version": "1.0.0",
            "intergrax_version": ">=1.0,<2",
            "documentation_uri": "sk-live-not-a-manifest-key",
        }
    )
    assert manifest.documentation_uri == "sk-live-not-a-manifest-key"


def test_secret_key_inside_capability_list_rejected() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="secret-like manifest field"):
        parse_platform_plugin_manifest_data(
            {
                "name": "acme-intergrax",
                "version": "1.0.0",
                "intergrax_version": ">=1.0,<2",
                "capabilities": [{"api_key": "x"}],
            }
        )


def test_parse_platform_plugin_pyproject_toml_valid() -> None:
    manifest = parse_platform_plugin_pyproject_toml(
        """
        [project]
        name = "acme-intergrax"
        version = "1.0.0"

        [tool.intergrax.plugin]
        name = "acme-intergrax"
        version = "1.0.0"
        intergrax_version = ">=1.0,<2"

        [[tool.intergrax.plugin.capabilities]]
        domain = "integrations"
        entry_point_group = "intergrax.integrations"
        entry_point_name = "acme_foo"
        """
    )
    assert manifest.package.name == "acme-intergrax"
    assert manifest.capabilities[0].entry_point_name == "acme_foo"


def test_parse_platform_plugin_pyproject_missing_table() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="missing \\[tool.intergrax.plugin\\]"):
        parse_platform_plugin_pyproject({"project": {"name": "acme-intergrax"}})


def test_parse_platform_plugin_pyproject_toml_malformed() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="invalid pyproject TOML"):
        parse_platform_plugin_pyproject_toml("[tool.intergrax.plugin\nname = broken")


def test_pyproject_normalized_project_name_matches_manifest() -> None:
    manifest = parse_platform_plugin_pyproject_toml(
        """
        [project]
        name = "Acme-Intergrax"
        version = "1.0.0"

        [tool.intergrax.plugin]
        name = "acme-intergrax"
        version = "1.0.0"
        intergrax_version = ">=1.0,<2"
        """
    )
    assert manifest.package.name == "acme-intergrax"
    assert manifest.package.version == "1.0.0"


def test_pyproject_name_conflicts_with_manifest() -> None:
    with pytest.raises(
        PlatformPluginManifestValidationError,
        match="manifest package name conflicts with \\[project\\].name",
    ):
        parse_platform_plugin_pyproject_toml(
            """
            [project]
            name = "acme-intergrax"
            version = "1.0.0"

            [tool.intergrax.plugin]
            name = "other-plugin"
            version = "1.0.0"
            intergrax_version = ">=1"
            """
        )


def test_pyproject_version_conflicts_with_manifest() -> None:
    with pytest.raises(
        PlatformPluginManifestValidationError,
        match="manifest package version conflicts with \\[project\\].version",
    ):
        parse_platform_plugin_pyproject_toml(
            """
            [project]
            name = "acme-intergrax"
            version = "1.0.0"

            [tool.intergrax.plugin]
            name = "acme-intergrax"
            version = "9.0.0"
            intergrax_version = ">=1"
            """
        )


def test_flat_and_nested_package_identity_identical() -> None:
    manifest = parse_platform_plugin_manifest_data(
        {
            "name": "acme-intergrax",
            "version": "1.0.0",
            "package": {"name": "Acme-Intergrax", "version": "1.0.0"},
            "intergrax_version": ">=1.0,<2",
        }
    )
    assert manifest.package.name == "acme-intergrax"
    assert manifest.package.version == "1.0.0"


def test_flat_and_nested_package_identity_contradictory() -> None:
    with pytest.raises(
        PlatformPluginManifestValidationError,
        match="conflicting manifest package name",
    ):
        parse_platform_plugin_manifest_data(
            {
                "name": "acme-intergrax",
                "version": "1.0.0",
                "package": {"name": "other-plugin", "version": "1.0.0"},
                "intergrax_version": ">=1.0,<2",
            }
        )


def test_flat_and_nested_compatibility_identical() -> None:
    manifest = parse_platform_plugin_manifest_data(
        {
            "name": "acme-intergrax",
            "version": "1.0.0",
            "intergrax_version": ">=1.0,<2",
            "platform_compatibility": {"intergrax_version": "<2,>=1.0"},
        }
    )
    assert manifest.platform_compatibility.intergrax_version == "<2,>=1.0"


def test_flat_and_nested_compatibility_contradictory() -> None:
    with pytest.raises(
        PlatformPluginManifestValidationError,
        match="conflicting manifest intergrax_version",
    ):
        parse_platform_plugin_manifest_data(
            {
                "name": "acme-intergrax",
                "version": "1.0.0",
                "intergrax_version": ">=1.0,<2",
                "platform_compatibility": {"intergrax_version": ">=2.0"},
            }
        )


def test_unsupported_schema_version_rejected() -> None:
    with pytest.raises(PlatformPluginManifestValidationError, match="unsupported manifest schema_version"):
        parse_platform_plugin_manifest_data(
            {
                "schema_version": 99,
                "name": "acme-intergrax",
                "version": "1.0.0",
                "intergrax_version": ">=1.0,<2",
            }
        )


def test_manifest_models_are_immutable() -> None:
    manifest = build_platform_plugin_manifest(
        name="acme-intergrax",
        version="1.0.0",
        intergrax_version=">=1.0,<2",
    )
    with pytest.raises(ValidationError):
        manifest.package = DistributionPackageIdentity(name="other", version="2.0.0")  # type: ignore[misc]


def test_manifest_construction_has_no_registration_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime registration must not occur during manifest parsing")

    monkeypatch.setattr(
        "intergrax.core.plugins.discovery.load_entry_point_plugins",
        _fail_if_called,
    )
    monkeypatch.setattr(
        "intergrax.core.plugins.discovery.register_plugins",
        _fail_if_called,
    )

    parse_platform_plugin_pyproject_toml(
        """
        [tool.intergrax.plugin]
        name = "acme-intergrax"
        version = "1.0.0"
        intergrax_version = ">=1.0,<2"
        """
    )
