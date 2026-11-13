# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Stable public hosting API manifest (APP-HOST-9F).

The public hosting API is the set of names exported by ``intergrax.hosting.__all__``.
This module declares that set, a structural fingerprint per symbol, and a version history
under the COMPAT-X versioning policy (the owning domain controls its version identity; a
breaking or unknown-impact change requires an explicit new version).

The manifest is mechanically enforced by
``tests/qualification/pdc_x/d4/test_d4_public_api_manifest.py``:

* an exported symbol that is not declared, or a declared symbol that is not exported, fails;
* a changed structural fingerprint of a declared symbol fails;
* the surface digest of the current version must match the recorded history, and a changed
  surface requires a new history entry with a higher version.

Regenerate the declaration after an intentional, reviewed change with::

    uv run python -m intergrax.hosting.public_api
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from typing import Final

HOSTING_PUBLIC_API_MANIFEST_SCHEMA_ID: Final[str] = "intergrax.hosting.public_api"

# Current public API version (semantic version owned by hosting). Bump when the declared
# surface below changes and add the matching entry to HOSTING_PUBLIC_API_VERSION_HISTORY.
HOSTING_PUBLIC_API_VERSION: Final[str] = "1.0.0"

_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")


def _stable(text: str) -> str:
    return _ADDRESS.sub("0x?", text)


def _signature_text(obj: object) -> str:
    try:
        return _stable(str(inspect.signature(obj)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "<no-signature>"


def _public_members(cls: type) -> dict[str, str]:
    members: dict[str, str] = {}
    for name, value in sorted(vars(cls).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, (staticmethod, classmethod)):
            members[name] = "static" if isinstance(value, staticmethod) else "class"
            members[name] += _signature_text(value.__func__)
        elif inspect.isfunction(value):
            members[name] = _signature_text(value)
        elif isinstance(value, property):
            members[name] = "property"
    return members


def describe_public_symbol(obj: object) -> dict[str, object]:
    """Return a deterministic structural description of one public symbol."""
    if inspect.isclass(obj):
        if issubclass(obj, enum.Enum):
            return {
                "kind": "enum",
                "members": [(member.name, repr(member.value)) for member in obj],
            }
        description: dict[str, object] = {
            "kind": "class",
            "bases": sorted(base.__name__ for base in obj.__bases__),
            "members": _public_members(obj),
        }
        model_fields = getattr(obj, "model_fields", None)
        if isinstance(model_fields, dict):
            description["kind"] = "model"
            description["fields"] = {
                name: [_stable(str(field.annotation)), field.is_required()]
                for name, field in sorted(model_fields.items())
            }
        elif dataclasses.is_dataclass(obj):
            description["kind"] = "dataclass"
            description["fields"] = {field.name: _stable(str(field.type)) for field in dataclasses.fields(obj)}
        elif getattr(obj, "_is_protocol", False):
            description["kind"] = "protocol"
        return description
    if callable(obj):
        return {"kind": "function", "signature": _signature_text(obj)}
    return {"kind": "constant", "value": _stable(repr(obj))}


def fingerprint_public_symbol(obj: object) -> str:
    payload = json.dumps(describe_public_symbol(obj), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_public_api_fingerprints() -> dict[str, str]:
    """Fingerprint every symbol currently exported by ``intergrax.hosting``."""
    import intergrax.hosting as hosting

    return {name: fingerprint_public_symbol(getattr(hosting, name)) for name in sorted(set(hosting.__all__))}


def compute_surface_digest(fingerprints: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(fingerprints.items())), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_manifest_declaration() -> str:
    """Render the declaration block for HOSTING_PUBLIC_API_FINGERPRINTS (maintainer tool)."""
    fingerprints = compute_public_api_fingerprints()
    lines = ["HOSTING_PUBLIC_API_FINGERPRINTS: Final[Mapping[str, str]] = {"]
    lines.extend(f'    "{name}": "{value}",' for name, value in fingerprints.items())
    lines.append("}")
    lines.append("")
    lines.append(f'surface digest: "{compute_surface_digest(fingerprints)}"')
    return "\n".join(lines)


_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


def _version_key(version: str) -> tuple[int, int, int]:
    major, minor, patch = (int(part) for part in version.split("."))
    return major, minor, patch


def verify_public_api(
    live: Mapping[str, str],
    declared: Mapping[str, str],
    history: Mapping[str, str],
    version: str,
) -> list[str]:
    """Return COMPAT-X version-policy violations for a live public surface (empty = compliant).

    Fail closed: any difference between the live surface and the declaration is a violation
    until the declaration and a new, higher version history entry are updated together.
    """
    violations: list[str] = []
    for name in sorted(set(live) - set(declared)):
        violations.append(f"undeclared public symbol: {name}")
    for name in sorted(set(declared) - set(live)):
        violations.append(f"declared public symbol is no longer exported: {name}")
    for name in sorted(set(live) & set(declared)):
        if live[name] != declared[name]:
            violations.append(f"public symbol changed without a declared version: {name}")

    if not _SEMVER.match(version):
        violations.append(f"public API version is not a semantic version: {version}")
        return violations
    bad_keys = [key for key in history if not _SEMVER.match(key)]
    violations.extend(f"history version is not a semantic version: {key}" for key in bad_keys)
    if bad_keys:
        return violations
    if version not in history:
        violations.append(f"current version {version} has no history entry")
    elif max(history, key=_version_key) != version:
        violations.append("current version is not the highest version in the history")
    digests = list(history.values())
    if len(set(digests)) != len(digests):
        violations.append("two public API versions record the same surface digest")
    recorded = history.get(version)
    if recorded is not None and recorded != compute_surface_digest(live):
        violations.append(
            f"live surface digest differs from the digest recorded for version {version}; "
            "bump the version and add a history entry"
        )
    if recorded is not None and recorded != compute_surface_digest(declared):
        violations.append(f"declared fingerprints differ from the digest recorded for version {version}")
    return violations


# Declared public surface: symbol name -> structural fingerprint.
HOSTING_PUBLIC_API_FINGERPRINTS: Final[Mapping[str, str]] = {
    "BOOTSTRAP_UNHANDLED_EXCEPTION_REASON_CODE": "4e69776a88fce721",
    "BootstrapIdentitySnapshot": "c1ebe66d17f54e3b",
    "BootstrapReadinessLevel": "c9dec6db5396ea9a",
    "BootstrapSurfaceKind": "32ef522a61966f6d",
    "ComponentFailurePolicy": "a305770868f5b83a",
    "CompositeHostedApplicationEventPublisher": "f193aa96746499fb",
    "FileHostedApplicationInstanceGuard": "e1315ff8c3879f8c",
    "HOSTED_APPLICATION_PROFILE_SPEC_VERSION": "900dfafb2497b467",
    "HookFailurePolicy": "07220ec76d614a0f",
    "HostedApplicationActiveWorkController": "a81294d185fcf42a",
    "HostedApplicationClock": "75695384cd7bb1ae",
    "HostedApplicationComponent": "49de0c11d9928c99",
    "HostedApplicationComponentError": "fadb43d6ffda51f5",
    "HostedApplicationComponentHealth": "df83960cb9faf322",
    "HostedApplicationComponentRegistration": "5c3ab2a6d0078bb4",
    "HostedApplicationComponentState": "492edda90c3869f7",
    "HostedApplicationConfigurationError": "fadb43d6ffda51f5",
    "HostedApplicationContext": "24fe1bf7ce68bf83",
    "HostedApplicationContextPublicView": "6f6c34e1dba49eb0",
    "HostedApplicationControlCoordinator": "c117f645181c4a09",
    "HostedApplicationControlError": "fadb43d6ffda51f5",
    "HostedApplicationControlIntent": "f8b3f3f9cc23b8ed",
    "HostedApplicationControlSnapshot": "f67219a2c1050d91",
    "HostedApplicationDefinition": "1301939e7f4c37ae",
    "HostedApplicationDefinitionError": "fadb43d6ffda51f5",
    "HostedApplicationDefinitionPublicView": "12596a22e51df227",
    "HostedApplicationDiagnosticSnapshot": "f0f2b87625c7a0e3",
    "HostedApplicationEffectiveControlRequest": "a77c1c0d2a0a55b5",
    "HostedApplicationEngine": "282ec2073f5931c9",
    "HostedApplicationEngineError": "1628b26193f04b65",
    "HostedApplicationEngineFactory": "476a39cafdf59241",
    "HostedApplicationEngineTerminalResult": "e723614f856ec83a",
    "HostedApplicationEvent": "6a43892d1df3a963",
    "HostedApplicationEventPublisher": "c80f749bed94e182",
    "HostedApplicationEventSubscription": "f33f0bb73cf19cd2",
    "HostedApplicationEventType": "d690229ce1d9b3a9",
    "HostedApplicationExitKind": "fbe0f571f33c07f4",
    "HostedApplicationExitRecord": "5c90bbb2a2e1e3af",
    "HostedApplicationFailurePhase": "d6f11a1c3e7c399e",
    "HostedApplicationFailureRecord": "28bf4739ea2e8875",
    "HostedApplicationFlushService": "3ee9348de98dbe29",
    "HostedApplicationHealthSnapshot": "48abde86c5fe54eb",
    "HostedApplicationHook": "1bcc42a5fdf47979",
    "HostedApplicationHookError": "fadb43d6ffda51f5",
    "HostedApplicationHookMode": "e21237c644366b58",
    "HostedApplicationHookPoint": "515184f069de341b",
    "HostedApplicationHooks": "265d5f9b1e59eeaf",
    "HostedApplicationIdentity": "10b421173055ced8",
    "HostedApplicationInstanceAcquisitionResult": "e90bf3d1b293d4e7",
    "HostedApplicationInstanceConflictError": "8f6ebe86101e62d8",
    "HostedApplicationInstanceGuardError": "fadb43d6ffda51f5",
    "HostedApplicationInstanceLeasePublicView": "ea1fe98864513ed9",
    "HostedApplicationInstanceOwnershipError": "8f6ebe86101e62d8",
    "HostedApplicationLifecycleMetric": "d567d568532153ba",
    "HostedApplicationLifecycleSnapshot": "781df96f37b93589",
    "HostedApplicationLifecycleSnapshotProvider": "2d9f168077ffaf17",
    "HostedApplicationLifecycleState": "b76dbce4cadd7dd8",
    "HostedApplicationLifecycleTransitionError": "fadb43d6ffda51f5",
    "HostedApplicationLogger": "279e85f9bbdb64af",
    "HostedApplicationPaths": "ceb9c290119140c2",
    "HostedApplicationProcessIdentity": "44cae10772de862d",
    "HostedApplicationProfile": "7e5dc1bf59f1124a",
    "HostedApplicationProfilePublicView": "744d9c9d8440b58d",
    "HostedApplicationReadinessService": "e06db16f5630289c",
    "HostedApplicationRestartDecision": "213e60f65a026f64",
    "HostedApplicationRestartPolicyError": "fadb43d6ffda51f5",
    "HostedApplicationRestartPolicyEvaluator": "2b3398e205790f28",
    "HostedApplicationRestartRequestSnapshot": "e0caa003eebd11f0",
    "HostedApplicationRuntime": "683497b0b40342a5",
    "HostedApplicationRuntimeError": "fadb43d6ffda51f5",
    "HostedApplicationServiceRegistry": "91e305a6adc483c3",
    "HostedApplicationShutdownCoordinator": "fde2e028189d8584",
    "HostedApplicationShutdownError": "fadb43d6ffda51f5",
    "HostedApplicationShutdownExecutionSnapshot": "9c749099ed1ed933",
    "HostedApplicationShutdownRequestSnapshot": "e0caa003eebd11f0",
    "HostedApplicationShutdownTimeoutError": "0fa77807521490e5",
    "HostedApplicationSignalError": "fadb43d6ffda51f5",
    "HostedApplicationStartupError": "fadb43d6ffda51f5",
    "HostedApplicationSupervisor": "f8f3e2d4e60e65b2",
    "HostedApplicationSupervisorError": "5d80a5e3f580f78c",
    "HostedApplicationSupervisorLaunchContext": "a1c9acc7c4113892",
    "HostedApplicationSupervisorResult": "b67e568a1d7acd4a",
    "HostedBootstrapFailureProducer": "78aeffa8b7de573f",
    "HostedProcessBootstrapContext": "fd90b265000c0480",
    "HostedProcessBootstrapFailureFacts": "dcdae04bd97e9dae",
    "HostedProcessBootstrapPhase": "79003fc89f85b138",
    "InstancePolicy": "4596b83dcf4eeefb",
    "LifecyclePolicy": "d00415d521a7e330",
    "LoggingBootstrapFailureReporter": "477a226bbd628f1e",
    "ObservabilityHostedApplicationEventPublisher": "f193aa96746499fb",
    "PortableForegroundSignalAdapter": "cc2ce40aeee449f1",
    "RestartPolicy": "a0fb20d6905c0ba5",
    "ShutdownPolicy": "82222717b3373c0f",
    "hosted_process_bootstrap_failure_payload": "710ec06fabbb195e",
    "mint_bootstrap_attempt_id": "6707085707a57420",
    "resolve_hosted_application_definition": "6a115995435bc66d",
    "run_guarded_hosted_bootstrap_segment": "978c080d0b02332a",
    "run_guarded_hosted_process_bootstrap": "91c6533937d33839",
    "run_hosted_application": "3ebfd66b564e0268",
}

# Version history. Each released public API version records the surface digest it declared.
# A changed surface must add a new, higher version here; recorded digests are never edited.
HOSTING_PUBLIC_API_VERSION_HISTORY: Final[Mapping[str, str]] = {
    "1.0.0": "sha256:88814bef45fe1494f1a2a5d22165c932063eba8a77df96db83bf59efb9134907",
}


if __name__ == "__main__":
    print(render_manifest_declaration())
