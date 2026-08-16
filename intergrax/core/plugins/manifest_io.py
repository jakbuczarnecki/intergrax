# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Parse optional Platform Plugin manifest sources (PLATFORM-PLUGIN-3)."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from intergrax.core.distribution import DistributionPackageIdentity, PlatformCompatibility
from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
from intergrax.core.plugins.package_contract import (
    MANIFEST_SCHEMA_VERSION,
    CapabilityDescriptor,
    PlatformPluginManifest,
    validate_platform_plugin_manifest_secrets,
)

_PYPROJECT_TOOL_PATH = ("tool", "intergrax", "plugin")
_ALLOWED_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "version",
        "intergrax_version",
        "package",
        "platform_compatibility",
        "capabilities",
        "author",
        "documentation_uri",
        "labels",
    }
)


def _validation_error(message: str, *, cause: Exception | None = None) -> PlatformPluginManifestValidationError:
    if cause is None:
        return PlatformPluginManifestValidationError(message)
    error = PlatformPluginManifestValidationError(message)
    error.__cause__ = cause
    return error


def _require_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise _validation_error(f"{field_name} must be a mapping")
    return value


def _reject_unknown_manifest_keys(payload: Mapping[str, Any]) -> None:
    unknown = sorted(set(payload) - _ALLOWED_MANIFEST_KEYS)
    if unknown:
        raise _validation_error(f"unknown manifest field(s): {', '.join(unknown)}")


def _coerce_package_identity(payload: Mapping[str, Any]) -> DistributionPackageIdentity:
    flat_name = payload.get("name")
    flat_version = payload.get("version")
    package_payload = payload.get("package")

    has_flat = flat_name is not None and flat_version is not None
    has_flat_partial = (flat_name is not None or flat_version is not None) and not has_flat
    if has_flat_partial:
        raise _validation_error(
            "manifest must include both name and version when using flat package identity"
        )

    has_nested = package_payload is not None
    if not has_flat and not has_nested:
        raise _validation_error("manifest must include package.name and package.version")

    if has_flat and has_nested:
        flat_identity = DistributionPackageIdentity(name=str(flat_name), version=str(flat_version))
        package_mapping = _require_mapping(package_payload, field_name="package")
        nested_identity = DistributionPackageIdentity.model_validate(package_mapping)
        if flat_identity.name != nested_identity.name:
            raise _validation_error(
                "conflicting manifest package name: flat name and package.name disagree "
                f"({flat_identity.name!r} vs {nested_identity.name!r})"
            )
        if flat_identity.version != nested_identity.version:
            raise _validation_error(
                "conflicting manifest package version: flat version and package.version disagree "
                f"({flat_identity.version!r} vs {nested_identity.version!r})"
            )
        return flat_identity

    if has_flat:
        return DistributionPackageIdentity(name=str(flat_name), version=str(flat_version))

    package_mapping = _require_mapping(package_payload, field_name="package")
    return DistributionPackageIdentity.model_validate(package_mapping)


def _coerce_platform_compatibility(payload: Mapping[str, Any]) -> PlatformCompatibility:
    flat_intergrax = payload.get("intergrax_version")
    compatibility_payload = payload.get("platform_compatibility")

    has_flat = flat_intergrax is not None
    has_nested = compatibility_payload is not None
    if not has_flat and not has_nested:
        raise _validation_error(
            "manifest must include platform_compatibility.intergrax_version or intergrax_version"
        )

    if has_flat and has_nested:
        flat_compatibility = PlatformCompatibility(intergrax_version=str(flat_intergrax))
        compatibility_mapping = _require_mapping(
            compatibility_payload,
            field_name="platform_compatibility",
        )
        nested_compatibility = PlatformCompatibility.model_validate(compatibility_mapping)
        if flat_compatibility.intergrax_version != nested_compatibility.intergrax_version:
            raise _validation_error(
                "conflicting manifest intergrax_version: flat intergrax_version and "
                "platform_compatibility.intergrax_version disagree "
                f"({flat_compatibility.intergrax_version!r} vs "
                f"{nested_compatibility.intergrax_version!r})"
            )
        return flat_compatibility

    if has_flat:
        return PlatformCompatibility(intergrax_version=str(flat_intergrax))

    compatibility_mapping = _require_mapping(
        compatibility_payload,
        field_name="platform_compatibility",
    )
    return PlatformCompatibility.model_validate(compatibility_mapping)


def _validate_project_identity_against_manifest(
    pyproject: Mapping[str, Any],
    manifest: PlatformPluginManifest,
) -> None:
    project = pyproject.get("project")
    if project is None:
        return
    project_mapping = _require_mapping(project, field_name="project")
    project_name = project_mapping.get("name")
    project_version = project_mapping.get("version")
    if project_name is None or project_version is None:
        return
    try:
        project_identity = DistributionPackageIdentity(
            name=str(project_name),
            version=str(project_version),
        )
    except ValidationError as exc:
        raise _validation_error("invalid [project] distribution identity") from exc

    if project_identity.name != manifest.package.name:
        raise _validation_error(
            "manifest package name conflicts with [project].name: "
            f"manifest has {manifest.package.name!r}, [project] has {project_identity.name!r}"
        )
    if project_identity.version != manifest.package.version:
        raise _validation_error(
            "manifest package version conflicts with [project].version: "
            f"manifest has {manifest.package.version!r}, [project] has {project_identity.version!r}"
        )


def _coerce_capabilities(value: object) -> tuple[CapabilityDescriptor, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise _validation_error("capabilities must be a list")
    descriptors: list[CapabilityDescriptor] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise _validation_error(f"capabilities[{index}] must be a mapping")
        try:
            descriptors.append(CapabilityDescriptor.model_validate(item))
        except ValidationError as exc:
            raise _validation_error(
                f"invalid capability descriptor at capabilities[{index}]"
            ) from exc
    return tuple(descriptors)


def _normalize_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    validate_platform_plugin_manifest_secrets(payload)
    _reject_unknown_manifest_keys(payload)

    schema_version = payload.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise _validation_error(
            f"unsupported manifest schema_version: {schema_version!r}; expected {MANIFEST_SCHEMA_VERSION}"
        )

    package = _coerce_package_identity(payload)
    compatibility = _coerce_platform_compatibility(payload)

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package": package,
        "platform_compatibility": compatibility,
        "capabilities": _coerce_capabilities(payload.get("capabilities")),
        "author": payload.get("author"),
        "documentation_uri": payload.get("documentation_uri"),
        "labels": payload.get("labels", ()),
    }


def parse_platform_plugin_manifest_data(payload: Mapping[str, Any]) -> PlatformPluginManifest:
    """Parse standalone manifest data without ``[project]`` context or installed packages.

    Package identity may appear as flat ``name``/``version`` or nested ``package``.
    When both representations are present they must be semantically equal.
    For a complete ``pyproject.toml``, use ``parse_platform_plugin_pyproject`` so
    ``[project].name``/``version`` are validated as authoritative distribution identity.
    """
    try:
        normalized = _normalize_manifest_payload(payload)
        return PlatformPluginManifest.model_validate(normalized)
    except PlatformPluginManifestValidationError:
        raise
    except ValidationError as exc:
        raise _validation_error("invalid Platform Plugin manifest") from exc
    except ValueError as exc:
        raise _validation_error(str(exc)) from exc


def parse_platform_plugin_pyproject(pyproject: Mapping[str, Any]) -> PlatformPluginManifest:
    """Extract and parse ``[tool.intergrax.plugin]`` from a pyproject mapping."""
    tool = pyproject.get("tool")
    if tool is None:
        raise _validation_error("missing [tool.intergrax.plugin] table")
    tool_mapping = _require_mapping(tool, field_name="tool")
    intergrax = tool_mapping.get("intergrax")
    if intergrax is None:
        raise _validation_error("missing [tool.intergrax.plugin] table")
    intergrax_mapping = _require_mapping(intergrax, field_name="tool.intergrax")
    plugin_table = intergrax_mapping.get("plugin")
    if plugin_table is None:
        raise _validation_error("missing [tool.intergrax.plugin] table")
    plugin_mapping = _require_mapping(plugin_table, field_name="tool.intergrax.plugin")
    manifest = parse_platform_plugin_manifest_data(plugin_mapping)
    _validate_project_identity_against_manifest(pyproject, manifest)
    return manifest


def parse_platform_plugin_pyproject_toml(source: str) -> PlatformPluginManifest:
    """Parse manifest content from a pyproject TOML string."""
    try:
        data = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise _validation_error("invalid pyproject TOML") from exc
    if not isinstance(data, dict):
        raise _validation_error("pyproject root must be a mapping")
    return parse_platform_plugin_pyproject(data)
