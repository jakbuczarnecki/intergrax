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
        profiles=profiles,
        proof_kind="test",
        command=ProofArgvCommand(executable="python", argv=("-c", "print('ok')")),
        platform_requirements=platform_requirements,
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )


def test_manifest_loads(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    assert manifest.schema_version == "intergrax.proof_manifest.v2"
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


_QUICKSTART_IDS = frozenset(
    {
        "LKW-PRODUCT-QUICKSTART-WINDOWS",
        "LKW-PRODUCT-QUICKSTART-LINUX",
        "LKW-PRODUCT-QUICKSTART-MACOS",
    }
)


def _quickstart_entry(
    manifest: IntergraxProofManifest, proof_id: str
) -> ProofManifestEntry:
    return next(entry for entry in manifest.entries if entry.proof_id == proof_id)


def test_product_quickstart_proof_ids_unique(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    ids = [entry.proof_id for entry in manifest.entries]
    assert len(ids) == len(set(ids))
    assert _QUICKSTART_IDS.issubset(set(ids))


@pytest.mark.parametrize(
    ("proof_id", "platform", "os_family", "wrapper_id"),
    [
        (
            "LKW-PRODUCT-QUICKSTART-WINDOWS",
            "windows",
            "windows",
            "windows_bat",
        ),
        (
            "LKW-PRODUCT-QUICKSTART-LINUX",
            "linux",
            "linux",
            "linux_sh",
        ),
        (
            "LKW-PRODUCT-QUICKSTART-MACOS",
            "macos",
            "macos",
            "macos_sh",
        ),
    ],
)
def test_product_quickstart_platform_requirements(
    repo_root: Path,
    proof_id: str,
    platform: str,
    os_family: str,
    wrapper_id: str,
) -> None:
    manifest = load_manifest(repo_root=repo_root)
    entry = _quickstart_entry(manifest, proof_id)
    assert entry.platform_requirements == frozenset({platform})
    assert entry.public_evidence_eligible is True
    assert entry.safety_class is ProofSafetyClass.LOCAL_MUTATING
    argv = entry.command.argv
    assert argv[:5] == (
        "run",
        "--project",
        "applications/local_workspace_application",
        "python",
        "applications/local_workspace_application/scripts/run-lkw-product-quickstart.py",
    )
    assert "--os-family" in argv
    assert "--wrapper-id" in argv
    assert argv[argv.index("--os-family") + 1] == os_family
    assert argv[argv.index("--wrapper-id") + 1] == wrapper_id
    assert not any(token.endswith((".bat", ".sh")) for token in argv)


def test_full_includes_current_platform_product_quickstart(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    for platform_family, proof_id in (
        ("windows", "LKW-PRODUCT-QUICKSTART-WINDOWS"),
        ("linux", "LKW-PRODUCT-QUICKSTART-LINUX"),
        ("macos", "LKW-PRODUCT-QUICKSTART-MACOS"),
    ):
        selected = select_proofs_from_manifest(
            manifest, profile=ProofProfile.FULL, platform_family=platform_family
        )
        ids = {entry.proof_id for entry in selected}
        assert proof_id in ids


def test_quick_excludes_product_quickstart(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    selected = select_proofs_from_manifest(
        manifest, profile=ProofProfile.QUICK, platform_family="windows"
    )
    ids = {entry.proof_id for entry in selected}
    assert ids.isdisjoint(_QUICKSTART_IDS)


_LKW_INDEXED_EVIDENCE_IDS = frozenset(
    {
        "LKW-WEB-URL-INDEXED-ASK",
        "LKW-HYBRID-ASK-INDEXED",
    }
)


@pytest.mark.parametrize("proof_id", sorted(_LKW_INDEXED_EVIDENCE_IDS))
def test_indexed_evidence_proofs_registered(repo_root: Path, proof_id: str) -> None:
    manifest = load_manifest(repo_root=repo_root)
    entry = next(item for item in manifest.entries if item.proof_id == proof_id)
    assert entry.public_evidence_eligible is True
    assert entry.safety_class is ProofSafetyClass.LOCAL_MUTATING
    assert ProofProfile.FULL in entry.profiles
    assert ProofProfile.LIVE in entry.profiles
    assert ProofProfile.QUICK not in entry.profiles
    assert "run-lkw-" in entry.command.argv[-1]


def test_quick_excludes_indexed_evidence_proofs(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    selected = select_proofs_from_manifest(
        manifest, profile=ProofProfile.QUICK, platform_family="windows"
    )
    ids = {entry.proof_id for entry in selected}
    assert ids.isdisjoint(_LKW_INDEXED_EVIDENCE_IDS)
