# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
    PlatformProofDescriptor,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import (
    DescriptorLoadError,
    descriptor_to_manifest_entry,
    load_descriptor,
    normalize_to_manifest_entry,
)
from scripts.proof.intergrax_proof_contracts import (
    EnvRequirementKind,
    ProofManifestEntry,
    ProofProfile,
    ProofSafetyClass,
)
from scripts.proof.intergrax_proof_manifest import load_manifest

_TOOLS_DESCRIPTOR_REL = (
    Path("platform_proofs")
    / "tools"
    / "iterative_sql_investigation"
    / PROOF_DESCRIPTOR_FILENAME
)
_TOOLS_PROOF_ID = "TOOLS-ITERATIVE-SQL-INVESTIGATION"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


@pytest.fixture
def tools_descriptor_path(repo_root: Path) -> Path:
    return repo_root / _TOOLS_DESCRIPTOR_REL


@pytest.fixture
def tools_static_manifest_entry(repo_root: Path) -> ProofManifestEntry:
    manifest = load_manifest(repo_root=repo_root)
    return next(
        entry for entry in manifest.entries if entry.proof_id == _TOOLS_PROOF_ID
    )


def test_tools_descriptor_parses(tools_descriptor_path: Path, repo_root: Path) -> None:
    descriptor = load_descriptor(tools_descriptor_path, repo_root=repo_root)
    assert descriptor.schema_version == PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION
    assert descriptor.proof_id == _TOOLS_PROOF_ID
    assert descriptor.domain == "tools"
    assert descriptor.package_version == "1.0.0"
    assert descriptor.evidence_schema == "intergrax.platform_proof_evidence.v1"
    assert descriptor.report_required is True


def test_tools_descriptor_normalizes_to_static_manifest_entry(
    tools_descriptor_path: Path,
    repo_root: Path,
    tools_static_manifest_entry: ProofManifestEntry,
) -> None:
    normalized = descriptor_to_manifest_entry(
        tools_descriptor_path, repo_root=repo_root
    )
    assert normalized == tools_static_manifest_entry


def test_unknown_schema_version_rejected(tmp_path: Path, repo_root: Path) -> None:
    package = tmp_path / "platform_proofs" / "tools" / "sample"
    package.mkdir(parents=True)
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": "intergrax.platform_proof_descriptor.v99",
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {
                    "executable": "python",
                    "argv": ("-c", "print('ok')"),
                },
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
                "public_evidence_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises((DescriptorLoadError, ValidationError)):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        PlatformProofDescriptor.model_validate(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
                "unexpected": True,
            }
        )


def test_duplicate_profiles_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        PlatformProofDescriptor.model_validate(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full", "full"],
                "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
            }
        )


def test_malformed_env_requirement_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
                "environment_requirements": [
                    {"kind": "ENV_PRESENT", "name": ""},
                ],
            }
        )


def test_invalid_safety_class_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
                "timeout_seconds": 60,
                "safety_class": "REMOTE_MUTATING",
            }
        )


def test_unsafe_proof_id_rejected() -> None:
    with pytest.raises(ValidationError, match="proof_id"):
        PlatformProofDescriptor.model_validate(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "../escape",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
            }
        )


def test_command_string_instead_of_argv_object_rejected() -> None:
    with pytest.raises(ValidationError):
        PlatformProofDescriptor.model_validate(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": "uv run python foo.py && rm -rf /",
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
            }
        )


def test_path_traversal_rejected(tmp_path: Path) -> None:
    package = tmp_path / "platform_proofs" / "tools" / "sample"
    package.mkdir(parents=True)
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {
                    "executable": "python",
                    "argv": ["../../outside.py"],
                },
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DescriptorLoadError, match="traverse"):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_missing_entrypoint_rejected(tmp_path: Path) -> None:
    package = tmp_path / "platform_proofs" / "tools" / "sample"
    package.mkdir(parents=True)
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {
                    "executable": "python",
                    "argv": ["platform_proofs/tools/sample/run_proof.py"],
                },
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DescriptorLoadError, match="missing declared entrypoint"):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_deterministic_normalization(
    tools_descriptor_path: Path,
    repo_root: Path,
) -> None:
    first = descriptor_to_manifest_entry(tools_descriptor_path, repo_root=repo_root)
    second = descriptor_to_manifest_entry(tools_descriptor_path, repo_root=repo_root)
    assert first == second
    assert first.profiles == frozenset({ProofProfile.FULL, ProofProfile.LIVE})
    assert first.safety_class is ProofSafetyClass.LOCAL_MUTATING
    assert first.timeout_seconds == 3600


def test_descriptor_contains_no_secret_values(tmp_path: Path) -> None:
    package = tmp_path / "platform_proofs" / "tools" / "sample"
    package.mkdir(parents=True)
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            {
                "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
                "proof_id": "TOOLS-SAMPLE",
                "title": "sample",
                "domain": "tools",
                "proof_kind": "sample",
                "package_version": "1.0.0",
                "profiles": ["full"],
                "command": {"executable": "python", "argv": ["-c", "print('ok')"]},
                "timeout_seconds": 60,
                "safety_class": "LOCAL_READ_ONLY",
                "password": "must-not-appear",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(DescriptorLoadError, match="secret field"):
        load_descriptor(descriptor_path, repo_root=tmp_path)


def test_env_requirements_declare_names_only(
    tools_descriptor_path: Path,
    repo_root: Path,
) -> None:
    descriptor = load_descriptor(tools_descriptor_path, repo_root=repo_root)
    for requirement in descriptor.environment_requirements:
        assert requirement.kind is EnvRequirementKind.ENV_PRESENT or requirement.kind in {
            EnvRequirementKind.COMMAND_AVAILABLE,
            EnvRequirementKind.DOCKER_AVAILABLE,
        }
        assert requirement.name
        assert "=" not in requirement.name


def test_normalize_ignores_environment_state(
    tools_descriptor_path: Path,
    repo_root: Path,
) -> None:
    descriptor = load_descriptor(tools_descriptor_path, repo_root=repo_root)
    entry = normalize_to_manifest_entry(
        descriptor,
        package_root=tools_descriptor_path.parent,
        repo_root=repo_root,
    )
    assert entry.proof_id == _TOOLS_PROOF_ID
