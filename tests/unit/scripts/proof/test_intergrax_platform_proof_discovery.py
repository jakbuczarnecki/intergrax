# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.proof.intergrax_platform_proof_descriptor import (
    PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
    PROOF_DESCRIPTOR_FILENAME,
)
from scripts.proof.intergrax_platform_proof_discovery import (
    DiscoveredPlatformProof,
    PlatformProofDiscoveryError,
    discover_platform_proof_descriptors,
    entries_semantically_equivalent,
    merge_static_and_discovered_entries,
)
from scripts.proof.intergrax_platform_proof_descriptor_loader import (
    descriptor_to_manifest_entry,
)
from scripts.proof.intergrax_proof_contracts import (
    ProofArgvCommand,
)
from scripts.proof.intergrax_proof_manifest import (
    ManifestLoadError,
    build_manifest_entries,
    load_manifest,
)

_TOOLS_PROOF_ID = "TOOLS-ITERATIVE-SQL-INVESTIGATION"
_TOOLS_DESCRIPTOR_REL = (
    Path("platform_proofs")
    / "tools"
    / "iterative_sql_investigation"
    / PROOF_DESCRIPTOR_FILENAME
)
_TEST_DOMAIN_ROOT = Path("platform_proofs") / "test_domain"
_EXAMPLE_PROOF_ID = "TEST-DOMAIN-EXAMPLE-PROOF"


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _minimal_descriptor_payload(
    *,
    proof_id: str,
    entrypoint: str,
    timeout_seconds: int = 60,
) -> dict[str, object]:
    return {
        "schema_version": PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION,
        "proof_id": proof_id,
        "title": f"{proof_id} title",
        "domain": "test_domain",
        "proof_kind": "example",
        "package_version": "1.0.0",
        "profiles": ["quick"],
        "command": {
            "executable": "python",
            "argv": [entrypoint],
        },
        "timeout_seconds": timeout_seconds,
        "safety_class": "LOCAL_READ_ONLY",
        "public_evidence_eligible": False,
    }


def _write_example_package(
    repo_root: Path,
    *,
    proof_id: str = _EXAMPLE_PROOF_ID,
    payload: dict[str, object] | None = None,
    run_proof_body: str = "raise RuntimeError('must not import during discovery')\n",
) -> Path:
    package = repo_root / _TEST_DOMAIN_ROOT / "example_proof"
    package.mkdir(parents=True, exist_ok=True)
    run_proof = package / "run_proof.py"
    run_proof.write_text(run_proof_body, encoding="utf-8")
    descriptor_path = package / PROOF_DESCRIPTOR_FILENAME
    descriptor_path.write_text(
        json.dumps(
            payload
            or _minimal_descriptor_payload(
                proof_id=proof_id,
                entrypoint="platform_proofs/test_domain/example_proof/run_proof.py",
            )
        ),
        encoding="utf-8",
    )
    return package


def _remove_test_domain(repo_root: Path) -> None:
    shutil.rmtree(repo_root / _TEST_DOMAIN_ROOT, ignore_errors=True)


def _empty_static_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.proof.intergrax_proof_manifest.build_manifest_entries",
        lambda: (),
    )


def test_discovers_tools_descriptor_recursively(repo_root: Path) -> None:
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    tools = next(
        item for item in discovered if item.manifest_entry.proof_id == _TOOLS_PROOF_ID
    )
    assert tools.descriptor_path == repo_root / _TOOLS_DESCRIPTOR_REL


def test_discovered_tools_normalizes_correctly(repo_root: Path) -> None:
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    tools = next(
        item for item in discovered if item.manifest_entry.proof_id == _TOOLS_PROOF_ID
    )
    expected = descriptor_to_manifest_entry(
        repo_root / _TOOLS_DESCRIPTOR_REL,
        repo_root=repo_root,
    )
    assert tools.manifest_entry == expected


def test_merged_manifest_contains_tools_exactly_once(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    tools_entries = [
        entry for entry in manifest.entries if entry.proof_id == _TOOLS_PROOF_ID
    ]
    assert len(tools_entries) == 1


def test_equivalent_static_tools_twin_not_duplicated(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    static_tools = next(
        entry
        for entry in build_manifest_entries()
        if entry.proof_id == _TOOLS_PROOF_ID
    )
    merged_tools = next(
        entry for entry in manifest.entries if entry.proof_id == _TOOLS_PROOF_ID
    )
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    descriptor_tools = next(
        item.manifest_entry
        for item in discovered
        if item.manifest_entry.proof_id == _TOOLS_PROOF_ID
    )
    assert entries_semantically_equivalent(static_tools, descriptor_tools)
    assert merged_tools == descriptor_tools
    assert len(manifest.entries) == len(build_manifest_entries())


def test_static_twin_conflicting_timeout_fails_manifest(repo_root: Path) -> None:
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    tools = next(item for item in discovered if item.manifest_entry.proof_id == _TOOLS_PROOF_ID)
    conflicting = DiscoveredPlatformProof(
        descriptor_path=tools.descriptor_path,
        descriptor=tools.descriptor,
        manifest_entry=tools.manifest_entry.model_copy(update={"timeout_seconds": 1}),
    )
    with pytest.raises(PlatformProofDiscoveryError, match="conflicts with static manifest entry"):
        merge_static_and_discovered_entries(
            build_manifest_entries(),
            tuple(
                conflicting if item.manifest_entry.proof_id == _TOOLS_PROOF_ID else item
                for item in discovered
            ),
            repo_root=repo_root,
        )


def test_static_twin_conflicting_argv_fails_manifest(repo_root: Path) -> None:
    discovered = discover_platform_proof_descriptors(repo_root=repo_root)
    tools = next(item for item in discovered if item.manifest_entry.proof_id == _TOOLS_PROOF_ID)
    conflicting = DiscoveredPlatformProof(
        descriptor_path=tools.descriptor_path,
        descriptor=tools.descriptor,
        manifest_entry=tools.manifest_entry.model_copy(
            update={
                "command": ProofArgvCommand(
                    executable="python",
                    argv=("-c", "print('conflict')"),
                )
            }
        ),
    )
    with pytest.raises(PlatformProofDiscoveryError, match="conflicts with static manifest entry"):
        merge_static_and_discovered_entries(
            build_manifest_entries(),
            tuple(
                conflicting if item.manifest_entry.proof_id == _TOOLS_PROOF_ID else item
                for item in discovered
            ),
            repo_root=repo_root,
        )


def test_two_discovered_descriptors_same_proof_id_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for slug in ("alpha", "beta"):
        package = tmp_path / "platform_proofs" / "test_domain" / slug
        package.mkdir(parents=True)
        (package / "run_proof.py").write_text("print('ok')\n", encoding="utf-8")
        (package / PROOF_DESCRIPTOR_FILENAME).write_text(
            json.dumps(
                _minimal_descriptor_payload(
                    proof_id="TEST-DUPLICATE",
                    entrypoint=f"platform_proofs/test_domain/{slug}/run_proof.py",
                )
            ),
            encoding="utf-8",
        )

    _empty_static_manifest(monkeypatch)
    with pytest.raises(ManifestLoadError, match="duplicate proof_id TEST-DUPLICATE"):
        load_manifest(repo_root=tmp_path)


def test_malformed_discovered_json_fails_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "platform_proofs" / "test_domain" / "broken"
    package.mkdir(parents=True)
    (package / "run_proof.py").write_text("print('ok')\n", encoding="utf-8")
    (package / PROOF_DESCRIPTOR_FILENAME).write_text("{not-json", encoding="utf-8")

    _empty_static_manifest(monkeypatch)
    with pytest.raises(ManifestLoadError, match="invalid platform proof descriptor"):
        load_manifest(repo_root=tmp_path)


def test_unknown_schema_discovered_descriptor_fails_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "platform_proofs" / "test_domain" / "broken"
    package.mkdir(parents=True)
    (package / "run_proof.py").write_text("print('ok')\n", encoding="utf-8")
    payload = _minimal_descriptor_payload(
        proof_id="TEST-UNKNOWN-SCHEMA",
        entrypoint="platform_proofs/test_domain/broken/run_proof.py",
    )
    payload["schema_version"] = "intergrax.platform_proof_descriptor.v99"
    (package / PROOF_DESCRIPTOR_FILENAME).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    _empty_static_manifest(monkeypatch)
    with pytest.raises(ManifestLoadError, match="schema_version"):
        load_manifest(repo_root=tmp_path)


def test_invalid_discovered_path_fails_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "platform_proofs" / "test_domain" / "broken"
    package.mkdir(parents=True)
    (package / PROOF_DESCRIPTOR_FILENAME).write_text(
        json.dumps(
            _minimal_descriptor_payload(
                proof_id="TEST-INVALID-PATH",
                entrypoint="platform_proofs/test_domain/broken/run_proof.py",
            )
        ),
        encoding="utf-8",
    )

    _empty_static_manifest(monkeypatch)
    with pytest.raises(ManifestLoadError, match="missing declared entrypoint"):
        load_manifest(repo_root=tmp_path)


def test_discovery_order_deterministic_independent_of_creation_order(
    tmp_path: Path,
) -> None:
    specs = (
        ("zulu", "TEST-ZULU"),
        ("alpha", "TEST-ALPHA"),
        ("mike", "TEST-MIKE"),
    )
    for slug, proof_id in reversed(specs):
        package = tmp_path / "platform_proofs" / "test_domain" / slug
        package.mkdir(parents=True)
        (package / "run_proof.py").write_text("print('ok')\n", encoding="utf-8")
        (package / PROOF_DESCRIPTOR_FILENAME).write_text(
            json.dumps(
                _minimal_descriptor_payload(
                    proof_id=proof_id,
                    entrypoint=f"platform_proofs/test_domain/{slug}/run_proof.py",
                )
            ),
            encoding="utf-8",
        )

    first = discover_platform_proof_descriptors(repo_root=tmp_path)
    second = discover_platform_proof_descriptors(repo_root=tmp_path)
    assert [item.manifest_entry.proof_id for item in first] == [
        item.manifest_entry.proof_id for item in second
    ]
    assert [item.manifest_entry.proof_id for item in first] == [
        "TEST-ALPHA",
        "TEST-MIKE",
        "TEST-ZULU",
    ]


def test_discovery_does_not_import_proof_modules(repo_root: Path) -> None:
    _remove_test_domain(repo_root)
    try:
        _write_example_package(
            repo_root,
            run_proof_body=(
                "import sys\n"
                "sys.stderr.write('IMPORT_SIDE_EFFECT\\n')\n"
                "raise RuntimeError('must not import during discovery')\n"
            ),
        )
        with patch.dict(sys.modules, {}, clear=False):
            discovered = discover_platform_proof_descriptors(repo_root=repo_root)
        assert any(
            item.manifest_entry.proof_id == _EXAMPLE_PROOF_ID for item in discovered
        )
    finally:
        _remove_test_domain(repo_root)


def test_discovery_does_not_execute_proof_entrypoint(repo_root: Path) -> None:
    _remove_test_domain(repo_root)
    try:
        _write_example_package(
            repo_root,
            run_proof_body="raise RuntimeError('must not execute during discovery')\n",
        )
        discover_platform_proof_descriptors(repo_root=repo_root)
    finally:
        _remove_test_domain(repo_root)


def test_static_non_platform_entries_remain(repo_root: Path) -> None:
    manifest = load_manifest(repo_root=repo_root)
    static_ids = {entry.proof_id for entry in build_manifest_entries()}
    merged_ids = {entry.proof_id for entry in manifest.entries}
    assert static_ids == merged_ids


def test_example_proof_self_registers_without_static_entry(repo_root: Path) -> None:
    _remove_test_domain(repo_root)
    try:
        _write_example_package(repo_root)
        manifest = load_manifest(repo_root=repo_root)
        assert any(
            entry.proof_id == _EXAMPLE_PROOF_ID for entry in manifest.entries
        )
        assert len(manifest.entries) == len(build_manifest_entries()) + 1
    finally:
        _remove_test_domain(repo_root)


def test_invalid_package_poisoning_manifest_fails_closed(repo_root: Path) -> None:
    _remove_test_domain(repo_root)
    try:
        _write_example_package(repo_root)
        broken = repo_root / _TEST_DOMAIN_ROOT / "broken"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / PROOF_DESCRIPTOR_FILENAME).write_text("{broken", encoding="utf-8")

        with pytest.raises(ManifestLoadError):
            load_manifest(repo_root=repo_root)
    finally:
        _remove_test_domain(repo_root)


def test_load_manifest_wraps_discovery_errors_as_manifest_load_error(
    repo_root: Path,
) -> None:
    _remove_test_domain(repo_root)
    try:
        broken = repo_root / _TEST_DOMAIN_ROOT / "broken"
        broken.mkdir(parents=True, exist_ok=True)
        (broken / PROOF_DESCRIPTOR_FILENAME).write_text("{broken", encoding="utf-8")

        with pytest.raises(ManifestLoadError) as exc_info:
            load_manifest(repo_root=repo_root)
        assert exc_info.type is ManifestLoadError
    finally:
        _remove_test_domain(repo_root)
