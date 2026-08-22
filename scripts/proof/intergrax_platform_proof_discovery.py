# © Artur Czarnecki. All rights reserved.

"""Discover descriptor-backed Platform Proof packages (PP-SUITE-2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.proof.intergrax_platform_proof_descriptor import (
    CANONICAL_PLATFORM_PROOF_ROOT,
    PROOF_DESCRIPTOR_FILENAME,
    PlatformProofDescriptor,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import (
    DescriptorLoadError,
    load_descriptor,
    normalize_to_manifest_entry,
)
from scripts.proof.intergrax_proof_contracts import ProofManifestEntry


class PlatformProofDiscoveryError(RuntimeError):
    """Hard failure during platform proof descriptor discovery."""


@dataclass(frozen=True, slots=True)
class DiscoveredPlatformProof:
    descriptor_path: Path
    descriptor: PlatformProofDescriptor
    manifest_entry: ProofManifestEntry


def discover_platform_proof_descriptors(
    *,
    repo_root: Path,
) -> tuple[DiscoveredPlatformProof, ...]:
    """Scan ``platform_proofs/`` for ``proof.json`` descriptors without importing proof code."""
    proofs_root = repo_root / CANONICAL_PLATFORM_PROOF_ROOT
    if not proofs_root.is_dir():
        return ()

    descriptor_paths = sorted(
        proofs_root.rglob(PROOF_DESCRIPTOR_FILENAME),
        key=lambda path: path.relative_to(repo_root).as_posix(),
    )

    discovered: list[DiscoveredPlatformProof] = []
    seen_proof_ids: dict[str, Path] = {}

    for descriptor_path in descriptor_paths:
        rel_path = _repo_relative_posix(descriptor_path, repo_root=repo_root)
        try:
            descriptor = load_descriptor(descriptor_path, repo_root=repo_root)
        except DescriptorLoadError as exc:
            raise PlatformProofDiscoveryError(
                f"invalid platform proof descriptor: {rel_path}: {exc}"
            ) from exc

        package_root = descriptor_path.parent
        manifest_entry = normalize_to_manifest_entry(
            descriptor,
            package_root=package_root,
            repo_root=repo_root,
        )

        if descriptor.proof_id in seen_proof_ids:
            previous = _repo_relative_posix(
                seen_proof_ids[descriptor.proof_id], repo_root=repo_root
            )
            raise PlatformProofDiscoveryError(
                f"duplicate proof_id {descriptor.proof_id}: {previous} and {rel_path}"
            )
        seen_proof_ids[descriptor.proof_id] = descriptor_path

        discovered.append(
            DiscoveredPlatformProof(
                descriptor_path=descriptor_path,
                descriptor=descriptor,
                manifest_entry=manifest_entry,
            )
        )

    return tuple(discovered)


def entries_semantically_equivalent(
    left: ProofManifestEntry,
    right: ProofManifestEntry,
) -> bool:
    """Compare execution-relevant manifest fields for migration-twin equivalence."""
    return left == right


def merge_static_and_discovered_entries(
    static_entries: tuple[ProofManifestEntry, ...],
    discovered: tuple[DiscoveredPlatformProof, ...],
    *,
    repo_root: Path,
) -> tuple[ProofManifestEntry, ...]:
    """Merge static manifest entries with discovered descriptor-backed proofs."""
    discovered_by_id = {
        item.manifest_entry.proof_id: item for item in discovered
    }
    consumed: set[str] = set()
    merged: list[ProofManifestEntry] = []

    for static_entry in static_entries:
        twin = discovered_by_id.get(static_entry.proof_id)
        if twin is None:
            merged.append(static_entry)
            continue

        if not entries_semantically_equivalent(static_entry, twin.manifest_entry):
            rel_path = _repo_relative_posix(twin.descriptor_path, repo_root=repo_root)
            raise PlatformProofDiscoveryError(
                f"platform proof descriptor conflicts with static manifest entry "
                f"{static_entry.proof_id}: {rel_path}"
            )

        merged.append(twin.manifest_entry)
        consumed.add(static_entry.proof_id)

    appended = sorted(
        (item for item in discovered if item.manifest_entry.proof_id not in consumed),
        key=lambda item: item.manifest_entry.proof_id,
    )
    merged.extend(item.manifest_entry for item in appended)
    return tuple(merged)


def _repo_relative_posix(path: Path, *, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()
