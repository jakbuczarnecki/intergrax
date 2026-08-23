# © Artur Czarnecki. All rights reserved.

"""Runner execution metadata for descriptor-backed Platform Proofs (PP-SUITE-3).

``ProofManifestEntry`` remains the subprocess execution contract.
``ProofExecutionSpec`` adds evidence policy without bloating manifest entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scripts.proof.intergrax_platform_proof_descriptor import ExpectedProofArtifact
from scripts.proof.intergrax_platform_proof_discovery import (
    DiscoveredPlatformProof,
    discover_platform_proof_descriptors,
    merge_static_and_discovered_entries,
)
from scripts.proof.intergrax_proof_contracts import (
    IntergraxProofManifest,
    ProofManifestEntry,
)
from scripts.proof.intergrax_proof_manifest import (
    ManifestLoadError,
    build_manifest_entries,
)

INTERGRAX_PROOF_ARTIFACT_DIR_ENV = "INTERGRAX_PROOF_ARTIFACT_DIR"


@dataclass(frozen=True, slots=True)
class ProofExecutionSpec:
    manifest_entry: ProofManifestEntry
    evidence_required: bool = False
    evidence_schema: str | None = None
    expected_artifacts: tuple[ExpectedProofArtifact, ...] = ()
    descriptor_path: Path | None = None
    package_root: Path | None = None
    expected_domains_exercised: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class LoadedManifestBundle:
    manifest: IntergraxProofManifest
    execution_specs: dict[str, ProofExecutionSpec]


def _spec_from_descriptor(
    entry: ProofManifestEntry,
    discovered: DiscoveredPlatformProof,
) -> ProofExecutionSpec:
    descriptor = discovered.descriptor
    return ProofExecutionSpec(
        manifest_entry=entry,
        evidence_required=descriptor.evidence_required,
        evidence_schema=descriptor.evidence_schema,
        expected_artifacts=descriptor.expected_artifacts,
        descriptor_path=discovered.descriptor_path,
        package_root=discovered.descriptor_path.parent,
        expected_domains_exercised=descriptor.domains_exercised,
    )


def _spec_without_evidence(entry: ProofManifestEntry) -> ProofExecutionSpec:
    return ProofExecutionSpec(manifest_entry=entry)


def build_execution_specs(
    entries: tuple[ProofManifestEntry, ...],
    discovered: tuple[DiscoveredPlatformProof, ...],
) -> dict[str, ProofExecutionSpec]:
    discovered_by_id = {item.manifest_entry.proof_id: item for item in discovered}
    specs: dict[str, ProofExecutionSpec] = {}
    for entry in entries:
        twin = discovered_by_id.get(entry.proof_id)
        if twin is not None:
            specs[entry.proof_id] = _spec_from_descriptor(entry, twin)
        else:
            specs[entry.proof_id] = _spec_without_evidence(entry)
    return specs


def load_manifest_bundle(*, repo_root: Path) -> LoadedManifestBundle:
    """Load manifest and per-proof execution metadata for evidence verification."""
    from pydantic import ValidationError

    from scripts.proof.intergrax_platform_proof_discovery import PlatformProofDiscoveryError
    from scripts.proof.intergrax_proof_manifest import _validate_entry_paths

    try:
        discovered = discover_platform_proof_descriptors(repo_root=repo_root)
        merged_entries = merge_static_and_discovered_entries(
            build_manifest_entries(),
            discovered,
            repo_root=repo_root,
        )
    except PlatformProofDiscoveryError as exc:
        raise ManifestLoadError(str(exc)) from exc

    try:
        manifest = IntergraxProofManifest(entries=merged_entries)
    except ValidationError as exc:
        raise ManifestLoadError(f"invalid manifest: {exc}") from exc

    for entry in manifest.entries:
        _validate_entry_paths(repo_root, entry)

    specs = build_execution_specs(manifest.entries, discovered)
    return LoadedManifestBundle(manifest=manifest, execution_specs=specs)


def suite_run_artifact_directory(repo_root: Path, suite_run_id: str) -> Path:
    return (repo_root / ".artifacts" / "proof" / suite_run_id).resolve()


def proof_run_artifact_directory(
    repo_root: Path,
    suite_run_id: str,
    proof_id: str,
) -> Path:
    return suite_run_artifact_directory(repo_root, suite_run_id) / "proofs" / proof_id
