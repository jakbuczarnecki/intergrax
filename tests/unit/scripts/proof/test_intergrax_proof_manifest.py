# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.proof.intergrax_proof_contracts import (
    IntergraxProofManifest,
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofSafetyClass,
)
from scripts.proof.intergrax_proof_manifest import (
    ManifestLoadError,
    expanded_profiles,
    load_manifest,
)
from scripts.proof.intergrax_proof_runner import select_proofs


def select_proofs_from_manifest(
    manifest: IntergraxProofManifest,
    *,
    profile: ProofProfile,
    platform_family: str,
):
    return select_proofs(manifest, profile=profile, platform_family=platform_family)


def _entry(
    proof_id: str,
    *,
    profiles: frozenset[ProofProfile],
    platform_requirements: frozenset[str] = frozenset(),
) -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        domain="test",
        profiles=profiles,
        proof_kind="test",
        command=ProofArgvCommand(executable="python", argv=("-c", "print('ok')")),
        platform_requirements=platform_requirements,
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )


def test_manifest_loads(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    assert manifest.schema_version == "intergrax.proof_manifest.v1"
    assert len(manifest.entries) >= 1


def test_duplicate_proof_id_rejected() -> None:
    entry = _entry("DUPLICATE", profiles=frozenset({ProofProfile.QUICK}))
    with pytest.raises(ValidationError, match="duplicate proof_id"):
        IntergraxProofManifest(entries=(entry, entry))


def test_empty_argv_rejected() -> None:
    with pytest.raises(ValidationError, match="argv must be non-empty"):
        ProofArgvCommand(executable="python", argv=())


def test_missing_declared_executable_rejected(tmp_path: Path) -> None:
    entry = ProofManifestEntry(
        proof_id="MISSING",
        title="missing",
        domain="test",
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="test",
        command=ProofArgvCommand(
            executable="python",
            argv=("scripts/does-not-exist.py",),
        ),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )
    manifest = IntergraxProofManifest(entries=(entry,))
    with pytest.raises(ManifestLoadError, match="missing declared executable"):
        for item in manifest.entries:
            from scripts.proof.intergrax_proof_manifest import _validate_entry_paths

            _validate_entry_paths(tmp_path, item)


def test_quick_selects_offline_token_proof(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    selected = select_proofs_from_manifest(
        manifest, profile=ProofProfile.QUICK, platform_family="windows"
    )
    ids = {entry.proof_id for entry in selected}
    assert "RUNTIME-TOKEN-OPTIMIZATION-OFFLINE" in ids
    assert "LKW-CORE-PLATFORM-WINDOWS" not in ids


def test_full_includes_quick_and_platform_specific(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    selected = select_proofs_from_manifest(
        manifest, profile=ProofProfile.FULL, platform_family="windows"
    )
    ids = {entry.proof_id for entry in selected}
    assert "RUNTIME-TOKEN-OPTIMIZATION-OFFLINE" in ids
    assert "LKW-CORE-PLATFORM-WINDOWS" in ids
    assert "LKW-OS-INTERACTION-MACOS" in ids


def test_live_includes_slack_proofs(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    selected = select_proofs_from_manifest(
        manifest, profile=ProofProfile.LIVE, platform_family="windows"
    )
    ids = {entry.proof_id for entry in selected}
    assert "SLACK-CONVERSATION-LIVE" in ids


def test_platform_filtering_at_runtime(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    selected = select_proofs_from_manifest(
        manifest, profile=ProofProfile.FULL, platform_family="windows"
    )
    macos_only = next(
        entry for entry in selected if entry.proof_id == "LKW-OS-INTERACTION-MACOS"
    )
    assert macos_only.platform_requirements == frozenset({"macos"})


def test_expanded_profiles_composition() -> None:
    assert expanded_profiles(ProofProfile.QUICK) == frozenset({ProofProfile.QUICK})
    assert expanded_profiles(ProofProfile.FULL) == frozenset(
        {ProofProfile.QUICK, ProofProfile.FULL}
    )
    assert expanded_profiles(ProofProfile.LIVE) == frozenset(
        {ProofProfile.QUICK, ProofProfile.FULL, ProofProfile.LIVE}
    )


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]
