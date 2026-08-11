# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Parse optional Platform Plugin manifest sources (PLATFORM-PLUGIN-3)."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
from intergrax.core.plugins.package_contract import (
    MANIFEST_SCHEMA_VERSION,
    CapabilityDescriptor,
    PlatformPluginManifest,
    PluginPackageIdentity,
    PlatformCompatibility,
    reject_secret_like_keys,
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
    reject_secret_like_keys(payload)
    _reject_unknown_manifest_keys(payload)

    schema_version = payload.get("schema_version", MANIFEST_SCHEMA_VERSION)
    if schema_version != MANIFEST_SCHEMA_VERSION:
        raise _validation_error(
            f"unsupported manifest schema_version: {schema_version!r}; expected {MANIFEST_SCHEMA_VERSION}"
        )

    package_payload = payload.get("package")
    if package_payload is None:
        if "name" not in payload or "version" not in payload:
            raise _validation_error("manifest must include package.name and package.version")
        package = PluginPackageIdentity(
            name=str(payload["name"]),
            version=str(payload["version"]),
        )
    else:
        package_mapping = _require_mapping(package_payload, field_name="package")
        package = PluginPackageIdentity.model_validate(package_mapping)

    compatibility_payload = payload.get("platform_compatibility")
    if compatibility_payload is None:
        if "intergrax_version" not in payload:
            raise _validation_error(
                "manifest must include platform_compatibility.intergrax_version or intergrax_version"
            )
        compatibility = PlatformCompatibility(intergrax_version=str(payload["intergrax_version"]))
    else:
        compatibility_mapping = _require_mapping(
            compatibility_payload,
            field_name="platform_compatibility",
        )
        compatibility = PlatformCompatibility.model_validate(compatibility_mapping)

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
    """Parse manifest data from a mapping without scanning installed packages."""
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
    return parse_platform_plugin_manifest_data(plugin_mapping)


def parse_platform_plugin_pyproject_toml(source: str) -> PlatformPluginManifest:
    """Parse manifest content from a pyproject TOML string."""
    try:
        data = tomllib.loads(source)
    except tomllib.TOMLDecodeError as exc:
        raise _validation_error("invalid pyproject TOML") from exc
    if not isinstance(data, dict):
        raise _validation_error("pyproject root must be a mapping")
    return parse_platform_plugin_pyproject(data)
