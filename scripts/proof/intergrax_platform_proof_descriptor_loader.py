# © Artur Czarnecki. All rights reserved.

"""Load and normalize Platform Proof package descriptors (PP-SUITE-1)."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from scripts.proof.intergrax_platform_proof_descriptor import (
    CANONICAL_PLATFORM_PROOF_ROOT,
    PROOF_DESCRIPTOR_FILENAME,
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PlatformProofDescriptor,
    _SECRET_FIELD_NAMES,
)
from scripts.proof.intergrax_proof_contracts import ProofManifestEntry, ProofProfile

_REPO_EXECUTABLE_SUFFIXES = (".py", ".toml")


class DescriptorLoadError(RuntimeError):
    """Hard failure loading or validating a platform proof descriptor."""


def load_descriptor(
    descriptor_path: Path,
    *,
    repo_root: Path,
) -> PlatformProofDescriptor:
    """Parse and validate ``proof.json`` without importing proof Python modules."""
    resolved = descriptor_path.resolve()
    if resolved.name != PROOF_DESCRIPTOR_FILENAME:
        raise DescriptorLoadError(
            f"descriptor must be named {PROOF_DESCRIPTOR_FILENAME}: {resolved}"
        )

    package_root = resolved.parent
    _assert_under_platform_proofs_root(resolved, repo_root=repo_root)

    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DescriptorLoadError(f"invalid JSON in {resolved}: {exc}") from exc

    _reject_secret_fields(payload, source=str(resolved))

    try:
        descriptor = PlatformProofDescriptor.model_validate(payload)
    except ValidationError as exc:
        raise DescriptorLoadError(
            f"invalid descriptor {resolved}: {exc}"
        ) from exc

    if descriptor.schema_version != PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION:
        raise DescriptorLoadError(
            f"unsupported schema_version in {resolved}: {descriptor.schema_version}"
        )

    _validate_command_paths(
        descriptor,
        package_root=package_root,
        repo_root=repo_root.resolve(),
    )
    return descriptor


def normalize_to_manifest_entry(
    descriptor: PlatformProofDescriptor,
    *,
    package_root: Path,
    repo_root: Path,
) -> ProofManifestEntry:
    """Deterministically map a validated descriptor to runner-facing manifest entry."""
    del package_root  # reserved for future package-relative path rules (PP-SUITE-2)
    del repo_root
    return ProofManifestEntry(
        proof_id=descriptor.proof_id,
        title=descriptor.title,
        domain=descriptor.domain,
        profiles=frozenset(descriptor.profiles),
        proof_kind=descriptor.proof_kind,
        command=descriptor.command,
        platform_requirements=frozenset(descriptor.platform_requirements),
        environment_requirements=descriptor.environment_requirements,
        external_provider=descriptor.external_provider,
        timeout_seconds=descriptor.timeout_seconds,
        safety_class=descriptor.safety_class,
        public_evidence_eligible=descriptor.public_evidence_eligible,
    )


def descriptor_to_manifest_entry(
    descriptor_path: Path,
    *,
    repo_root: Path,
) -> ProofManifestEntry:
    """Load descriptor and normalize to ``ProofManifestEntry``."""
    resolved = descriptor_path.resolve()
    descriptor = load_descriptor(resolved, repo_root=repo_root)
    return normalize_to_manifest_entry(
        descriptor,
        package_root=resolved.parent,
        repo_root=repo_root.resolve(),
    )


def _assert_under_platform_proofs_root(descriptor_path: Path, *, repo_root: Path) -> None:
    proofs_root = (repo_root / CANONICAL_PLATFORM_PROOF_ROOT).resolve()
    try:
        descriptor_path.resolve().relative_to(proofs_root)
    except ValueError as exc:
        raise DescriptorLoadError(
            f"descriptor must live under {CANONICAL_PLATFORM_PROOF_ROOT}/: {descriptor_path}"
        ) from exc


def _reject_secret_fields(payload: object, *, source: str) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = str(key).strip().lower()
            if normalized_key in _SECRET_FIELD_NAMES:
                raise DescriptorLoadError(
                    f"descriptor must not contain secret field {key!r} in {source}"
                )
            _reject_secret_fields(value, source=source)
        return
    if isinstance(payload, list):
        for item in payload:
            _reject_secret_fields(item, source=source)


def _validate_command_paths(
    descriptor: PlatformProofDescriptor,
    *,
    package_root: Path,
    repo_root: Path,
) -> None:
    entrypoints: list[Path] = []
    for token in descriptor.command.argv:
        normalized = token.replace("\\", "/")
        if _is_absolute_path(normalized):
            raise DescriptorLoadError(
                f"{descriptor.proof_id}: command argv must use repo-relative paths, "
                f"not absolute: {token!r}"
            )
        if ".." in PurePosixPath(normalized).parts:
            raise DescriptorLoadError(
                f"{descriptor.proof_id}: command argv must not traverse parents: {token!r}"
            )
        if normalized.endswith(_REPO_EXECUTABLE_SUFFIXES):
            resolved = (repo_root / normalized).resolve()
            try:
                resolved.relative_to(repo_root)
            except ValueError as exc:
                raise DescriptorLoadError(
                    f"{descriptor.proof_id}: command path escapes repository: {token!r}"
                ) from exc
            if not resolved.is_file():
                raise DescriptorLoadError(
                    f"{descriptor.proof_id}: missing declared entrypoint {token!r}"
                )
            entrypoints.append(resolved)

    if not entrypoints:
        raise DescriptorLoadError(
            f"{descriptor.proof_id}: command must declare at least one "
            f"{', '.join(_REPO_EXECUTABLE_SUFFIXES)} entrypoint"
        )

    package_resolved = package_root.resolve()
    if not any(
        _path_is_under(entrypoint, package_resolved) for entrypoint in entrypoints
    ):
        raise DescriptorLoadError(
            f"{descriptor.proof_id}: proof entrypoint must resolve inside package root "
            f"{package_resolved}"
        )


def _is_absolute_path(path: str) -> bool:
    if path.startswith("/"):
        return True
    return len(path) >= 2 and path[1] == ":"


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
