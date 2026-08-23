# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
    PlatformProofDescriptor,
)
from scripts.proof.intergrax_platform_proof_discovery import DiscoveredPlatformProof
from scripts.proof.intergrax_platform_proof_execution import (
    ProofExecutionSpec,
    build_execution_specs,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import (
    normalize_to_manifest_entry,
)
from scripts.proof.intergrax_proof_contracts import (
    ProofArgvCommand,
    ProofManifestEntry,
    ProofProfile,
    ProofSafetyClass,
)

pytestmark = pytest.mark.unit


def _minimal_descriptor_payload(
    *,
    domains_exercised: list[str] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "library_class": "CONFORMANCE",
        "proof_id": "TEST-EXEC-SPEC",
        "title": "title",
        "domains_exercised": domains_exercised or ["EXECUTION", "TOOLS"],
        "proof_kind": "example",
        "mechanisms_exercised": ["tools.sample_mechanism"],
        "package_version": "1.0.0",
        "profiles": ["quick"],
        "command": {
            "executable": "python",
            "argv": ["platform_proofs/test_domain/example/run_proof.py"],
        },
        "timeout_seconds": 60,
        "safety_class": "LOCAL_READ_ONLY",
    }


def _static_entry(proof_id: str = "STATIC-ONLY") -> ProofManifestEntry:
    return ProofManifestEntry(
        proof_id=proof_id,
        title=proof_id,
        profiles=frozenset({ProofProfile.QUICK}),
        proof_kind="static",
        command=ProofArgvCommand(executable="python", argv=("run.py",)),
        safety_class=ProofSafetyClass.LOCAL_READ_ONLY,
    )


def test_descriptor_backed_spec_propagates_expected_domains(tmp_path: Path) -> None:
    package = tmp_path / "platform_proofs" / "test_domain" / "example"
    package.mkdir(parents=True)
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            _minimal_descriptor_payload(
                domains_exercised=["TOOLS", "EXECUTION", "OBSERVABILITY"]
            )
        ),
        encoding="utf-8",
    )
    descriptor = PlatformProofDescriptor.model_validate(
        json.loads(descriptor_path.read_text(encoding="utf-8"))
    )
    entry = normalize_to_manifest_entry(
        descriptor,
        package_root=package,
        repo_root=tmp_path,
    )
    discovered = DiscoveredPlatformProof(
        manifest_entry=entry,
        descriptor=descriptor,
        descriptor_path=descriptor_path,
    )
    specs = build_execution_specs((entry,), (discovered,))
    spec = specs[entry.proof_id]
    assert spec.expected_domains_exercised == ("EXECUTION", "OBSERVABILITY", "TOOLS")
    assert spec.expected_domains_exercised == descriptor.domains_exercised


def test_static_manifest_spec_has_no_expected_domains() -> None:
    entry = _static_entry()
    specs = build_execution_specs((entry,), ())
    spec = specs[entry.proof_id]
    assert isinstance(spec, ProofExecutionSpec)
    assert spec.expected_domains_exercised is None
