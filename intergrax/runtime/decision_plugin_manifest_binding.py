# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Manifest capability binding for Decision platform plugins (metadata-only)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import EntryPointSpec
from intergrax.core.plugins.package_contract import CapabilityDescriptor


class ManifestCapabilityBindingDisposition(StrEnum):
    VALID = "valid"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class ManifestCapabilityBindingResult:
    disposition: ManifestCapabilityBindingDisposition
    rejection: PluginAdmissionRejection | None
    descriptor: CapabilityDescriptor | None = None


def _manifest_binding_rejected(
    spec: EntryPointSpec,
    *,
    reason_code: PluginAdmissionReasonCode,
    reason: str,
) -> ManifestCapabilityBindingResult:
    return ManifestCapabilityBindingResult(
        disposition=ManifestCapabilityBindingDisposition.REJECTED,
        rejection=PluginAdmissionRejection(
            spec=spec,
            reason_code=reason_code,
            reason=reason,
            fail_closed=True,
        ),
        descriptor=None,
    )


def _manifest_binding_valid(descriptor: CapabilityDescriptor) -> ManifestCapabilityBindingResult:
    return ManifestCapabilityBindingResult(
        disposition=ManifestCapabilityBindingDisposition.VALID,
        rejection=None,
        descriptor=descriptor,
    )


def validate_manifest_capability_binding(
    spec: EntryPointSpec,
    *,
    domain: str,
    capability_id: str,
) -> ManifestCapabilityBindingResult:
    if spec.distribution is None:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.UNRESOLVED_PACKAGE_IDENTITY,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: entry-point distribution identity "
                "is missing."
            ),
        )

    from importlib.metadata import PackageNotFoundError, distribution

    from intergrax.core.plugins.errors import PlatformPluginManifestValidationError
    from intergrax.core.plugins.manifest_io import parse_platform_plugin_pyproject_toml

    try:
        installed = distribution(spec.distribution)
    except PackageNotFoundError:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: distribution "
                f"{spec.distribution!r} is not installed or resolvable."
            ),
        )

    if installed.files is None:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: distribution "
                f"{spec.distribution!r} has no inspectable file metadata."
            ),
        )

    try:
        source = installed.read_text("pyproject.toml")
    except (FileNotFoundError, OSError, TypeError) as exc:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: Platform Plugin manifest for "
                f"distribution {spec.distribution!r} is unavailable ({type(exc).__name__})."
            ),
        )

    if source is None:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_BINDING_UNAVAILABLE,
            reason=(
                f"Manifest capability binding for entry point {spec.name!r} in group "
                f"{spec.group!r} cannot be proven: Platform Plugin manifest for "
                f"distribution {spec.distribution!r} is unavailable."
            ),
        )

    try:
        manifest = parse_platform_plugin_pyproject_toml(source)
    except PlatformPluginManifestValidationError:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_INVALID,
            reason=(
                f"Platform plugin manifest for distribution {spec.distribution!r} "
                f"is invalid or incomplete for entry point {spec.name!r} in group "
                f"{spec.group!r}."
            ),
        )

    if not manifest.capabilities:
        return _manifest_binding_rejected(
            spec,
            reason_code=PluginAdmissionReasonCode.MANIFEST_CAPABILITY_BINDING_MISSING,
            reason=(
                f"Platform plugin manifest for distribution {spec.distribution!r} "
                f"declares no capabilities for required binding "
                f"{capability_id!r} on entry point {spec.name!r} in group "
                f"{spec.group!r}."
            ),
        )

    for descriptor in manifest.capabilities:
        if (
            descriptor.domain == domain
            and descriptor.entry_point_group == spec.group
            and descriptor.entry_point_name == spec.name
        ):
            if capability_id not in descriptor.capability_ids:
                return _manifest_binding_rejected(
                    spec,
                    reason_code=PluginAdmissionReasonCode.CAPABILITY_ID_MISMATCH,
                    reason=(
                        f"Manifest capability_ids for entry point {spec.name!r} in group "
                        f"{spec.group!r} do not declare required capability "
                        f"{capability_id!r}."
                    ),
                )
            return _manifest_binding_valid(descriptor)

    return _manifest_binding_rejected(
        spec,
        reason_code=PluginAdmissionReasonCode.MANIFEST_CAPABILITY_BINDING_MISSING,
        reason=(
            f"Entry point {spec.name!r} in group {spec.group!r} is not declared "
            f"in the installed package Platform Plugin manifest capabilities for "
            f"required capability {capability_id!r}."
        ),
    )
