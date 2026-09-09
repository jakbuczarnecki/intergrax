"""VPI Data Pack distribution architecture and artifact contract tests."""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

from intergrax.proof_data import (
    DataPackageCache,
    DataPackageInstaller,
    DataPackageInstallRequest,
    DataPackageIntegrityError,
    HttpDataPackageTransport,
    LocalFileDataPackageTransport,
    PublicationStatus,
    load_proof_data_package_descriptor,
)
from intergrax.proof_data.descriptor import ProofDataPackageDescriptor
from platform_proofs.scenarios.verified_product_identification.data_package.build_descriptor import (
    build_data_pack_descriptor,
    build_descriptor_from_files,
    descriptor_identity,
)
from platform_proofs.scenarios.verified_product_identification.data_package.config import (
    VpiDataPackageConfig,
)
from platform_proofs.scenarios.verified_product_identification.data_package.content_policy import (
    BUILD_STATE_EXCLUSION_REASON,
    DISTRIBUTION_STRATEGY,
    collect_distributable_files,
)
from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
)
from platform_proofs.scenarios.verified_product_identification.data_package.install import (
    install_vpi_data_pack_distribution,
)
from platform_proofs.scenarios.verified_product_identification.data_package.publication import (
    assert_publication_permitted,
    is_publication_approved,
)
from platform_proofs.scenarios.verified_product_identification.data_package.summary import (
    summarize_descriptor,
)
from platform_proofs.scenarios.verified_product_identification.data_package.validation_handoff import (
    load_descriptor_validation_gate,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.contracts import (
    DataPackValidationVerdict,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.report import (
    write_validation_report_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.service import (
    validate_full_data_pack,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_full_validation_test_support import (
    build_valid_validation_fixture,
)

pytestmark = pytest.mark.unit

_DATA_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[5]
    / "platform_proofs"
    / "scenarios"
    / "verified_product_identification"
    / "data_package"
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def _validated_fixture(tmp_path: Path, monkeypatch):
    fixture = build_valid_validation_fixture(tmp_path / "fixture", monkeypatch)
    report = validate_full_data_pack(
        fixture.artifact_root,
        expectations=fixture.expectations,
        scratch_root=tmp_path / "scratch",
    )
    assert report.verdict is DataPackValidationVerdict.PASS
    report_path = tmp_path / "validation-report.json"
    write_validation_report_json(report_path, report)
    return fixture, report_path


def _build_descriptor(
    tmp_path: Path, monkeypatch
) -> tuple[ProofDataPackageDescriptor, Path, object, Path]:
    fixture, report_path = _validated_fixture(tmp_path, monkeypatch)
    output_path = tmp_path / "package.json"
    descriptor = build_data_pack_descriptor(
        artifact_root=fixture.artifact_root,
        validation_report_path=report_path,
        output_path=output_path,
        description="fixture data pack",
        provenance_ref="evidence/proof-report.json",
        redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        package_version="fixture-v1",
    )
    return descriptor, fixture.artifact_root, fixture, output_path


def _mirror_artifact(artifact_root: Path, mirror_root: Path) -> None:
    mirror_root.mkdir(parents=True, exist_ok=True)
    for path in artifact_root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(artifact_root)
            destination = mirror_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def test_descriptor_pass_from_validated_fixture(tmp_path: Path, monkeypatch) -> None:
    descriptor, _, fixture, _ = _build_descriptor(tmp_path, monkeypatch)
    summary = summarize_descriptor(descriptor)
    assert summary.total_files >= 5
    assert summary.relational_file_count == fixture.expectations.shard_count
    assert summary.embedding_file_count == fixture.expectations.shard_count
    assert summary.manifest_file_count == 1
    assert summary.checksum_file_count == 1
    assert summary.evidence_file_count == 1


def test_descriptor_rejects_partial_artifact(tmp_path: Path, monkeypatch) -> None:
    fixture, report_path = _validated_fixture(tmp_path, monkeypatch)
    paths = resolve_data_pack_paths(fixture.artifact_root)
    paths.manifest_file.unlink()
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        build_data_pack_descriptor(
            artifact_root=fixture.artifact_root,
            validation_report_path=report_path,
            output_path=tmp_path / "package.json",
            description="fixture",
            provenance_ref="evidence/proof-report.json",
            redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        )


def test_descriptor_rejects_missing_manifest(tmp_path: Path, monkeypatch) -> None:
    fixture, report_path = _validated_fixture(tmp_path, monkeypatch)
    resolve_data_pack_paths(fixture.artifact_root).manifest_file.unlink()
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        build_data_pack_descriptor(
            artifact_root=fixture.artifact_root,
            validation_report_path=report_path,
            output_path=tmp_path / "package.json",
            description="fixture",
            provenance_ref="evidence/proof-report.json",
            redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        )


def test_descriptor_rejects_validation_fail(tmp_path: Path, monkeypatch) -> None:
    fixture, report_path = _validated_fixture(tmp_path, monkeypatch)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["verdict"] = DataPackValidationVerdict.FAIL.value
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        build_data_pack_descriptor(
            artifact_root=fixture.artifact_root,
            validation_report_path=report_path,
            output_path=tmp_path / "package.json",
            description="fixture",
            provenance_ref="evidence/proof-report.json",
            redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        )


def test_descriptor_rejects_missing_file(tmp_path: Path, monkeypatch) -> None:
    fixture, report_path = _validated_fixture(tmp_path, monkeypatch)
    paths = resolve_data_pack_paths(fixture.artifact_root)
    paths.proof_report_file.unlink()
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        build_data_pack_descriptor(
            artifact_root=fixture.artifact_root,
            validation_report_path=report_path,
            output_path=tmp_path / "package.json",
            description="fixture",
            provenance_ref="evidence/proof-report.json",
            redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        )


def test_descriptor_rejects_checksum_mismatch(tmp_path: Path, monkeypatch) -> None:
    fixture, report_path = _validated_fixture(tmp_path, monkeypatch)
    paths = resolve_data_pack_paths(fixture.artifact_root)
    lines = paths.checksums_file.read_text(encoding="utf-8").splitlines()
    lines[0] = f"{'0' * 64}  {lines[0].split(maxsplit=1)[1]}"
    paths.checksums_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        collect_distributable_files(fixture.artifact_root)


def test_descriptor_rejects_duplicate_relative_path(tmp_path: Path) -> None:
    file_a = tmp_path / "a.bin"
    file_b = tmp_path / "b.bin"
    file_a.write_bytes(b"a")
    file_b.write_bytes(b"b")
    with pytest.raises(Exception):
        build_descriptor_from_files(
            files=(
                ("dup/path.bin", "ROLE", file_a),
                ("dup/path.bin", "ROLE", file_b),
            ),
            description="dup test",
            provenance_ref="provenance.json",
            redistribution_status=PublicationStatus.INTERNAL_BUILD,
            output_path=tmp_path / "package.json",
        )


def test_descriptor_rejects_unsafe_relative_path(tmp_path: Path) -> None:
    file_path = tmp_path / "unsafe.bin"
    file_path.write_bytes(b"x")
    with pytest.raises(Exception):
        build_descriptor_from_files(
            files=(("../escape.bin", "ROLE", file_path),),
            description="unsafe",
            provenance_ref="provenance.json",
            redistribution_status=PublicationStatus.INTERNAL_BUILD,
            output_path=tmp_path / "package.json",
        )


def test_descriptor_deterministic_ordering(tmp_path: Path, monkeypatch) -> None:
    descriptor, _, _, _ = _build_descriptor(tmp_path, monkeypatch)
    relative_paths = [entry.relative_path for entry in descriptor.files]
    assert relative_paths == sorted(relative_paths)


def test_descriptor_deterministic_output(tmp_path: Path, monkeypatch) -> None:
    descriptor_a, artifact_root, fixture, output_a = _build_descriptor(tmp_path / "a", monkeypatch)
    report_path = tmp_path / "a" / "validation-report.json"
    report = validate_full_data_pack(
        artifact_root,
        expectations=fixture.expectations,
        scratch_root=tmp_path / "a" / "scratch",
    )
    write_validation_report_json(report_path, report)
    output_b = tmp_path / "b" / "package.json"
    descriptor_b = build_data_pack_descriptor(
        artifact_root=artifact_root,
        validation_report_path=report_path,
        output_path=output_b,
        description="fixture data pack",
        provenance_ref="evidence/proof-report.json",
        redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        package_version="fixture-v1",
    )
    assert descriptor_identity(descriptor_a) == descriptor_identity(descriptor_b)
    assert output_b.read_text(encoding="utf-8") == output_a.read_text(encoding="utf-8")


def test_descriptor_identity_independent_of_base_uri(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, fixture, _ = _build_descriptor(tmp_path, monkeypatch)
    report_path = tmp_path / "validation-report.json"
    report = validate_full_data_pack(
        artifact_root,
        expectations=fixture.expectations,
        scratch_root=tmp_path / "scratch",
    )
    write_validation_report_json(report_path, report)
    descriptor_mirror_a = build_data_pack_descriptor(
        artifact_root=artifact_root,
        validation_report_path=report_path,
        output_path=tmp_path / "mirror-a.json",
        description="fixture data pack",
        provenance_ref="evidence/proof-report.json",
        redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        package_version="fixture-v1",
    )
    descriptor_mirror_b = build_data_pack_descriptor(
        artifact_root=artifact_root,
        validation_report_path=report_path,
        output_path=tmp_path / "mirror-b.json",
        description="fixture data pack",
        provenance_ref="evidence/proof-report.json",
        redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
        package_version="fixture-v1",
    )
    assert descriptor_identity(descriptor) == descriptor_identity(descriptor_mirror_a)
    assert descriptor_identity(descriptor_mirror_a) == descriptor_identity(descriptor_mirror_b)


def test_descriptor_identity_changes_when_bytes_change(tmp_path: Path, monkeypatch) -> None:
    descriptor_before, artifact_root, fixture, _ = _build_descriptor(tmp_path, monkeypatch)
    identity_before = descriptor_identity(descriptor_before)
    paths = resolve_data_pack_paths(artifact_root)
    paths.manifest_file.write_bytes(paths.manifest_file.read_bytes() + b" ")
    report_path = tmp_path / "validation-report.json"
    report = validate_full_data_pack(
        artifact_root,
        expectations=fixture.expectations,
        scratch_root=tmp_path / "scratch",
    )
    write_validation_report_json(report_path, report)
    if report.verdict is DataPackValidationVerdict.PASS:
        descriptor_after = build_data_pack_descriptor(
            artifact_root=artifact_root,
            validation_report_path=report_path,
            output_path=tmp_path / "changed.json",
            description="fixture data pack",
            provenance_ref="evidence/proof-report.json",
            redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
            package_version="fixture-v1",
        )
        assert descriptor_identity(descriptor_after) != identity_before
    else:
        with pytest.raises(VpiDataPackageDescriptorBuildError):
            build_data_pack_descriptor(
                artifact_root=artifact_root,
                validation_report_path=report_path,
                output_path=tmp_path / "changed.json",
                description="fixture data pack",
                provenance_ref="evidence/proof-report.json",
                redistribution_status=PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED,
                package_version="fixture-v1",
            )


def test_local_transport_full_install(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, fixture, descriptor_path = _build_descriptor(tmp_path, monkeypatch)
    mirror_root = tmp_path / "mirror"
    _mirror_artifact(artifact_root, mirror_root)
    config = VpiDataPackageConfig(
        descriptor_path=descriptor_path,
        install_dir=tmp_path / "installed",
        cache_dir=tmp_path / "cache",
        base_uri=None,
        package_version=descriptor.package_version,
    )
    result = install_vpi_data_pack_distribution(
        config,
        local_mirror_root=mirror_root,
    )
    assert result.install_report.verification_passed is True
    assert result.validation_report.manifest_status == "READY"
    assert result.validation_report.relational_shard_count == fixture.expectations.shard_count


def test_cache_reuse_on_second_install(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, fixture, descriptor_path = _build_descriptor(tmp_path, monkeypatch)
    mirror_root = tmp_path / "mirror"
    _mirror_artifact(artifact_root, mirror_root)
    cache_dir = tmp_path / "cache"
    config_a = VpiDataPackageConfig(
        descriptor_path=descriptor_path,
        install_dir=tmp_path / "installed-a",
        cache_dir=cache_dir,
        base_uri=None,
        package_version=descriptor.package_version,
    )
    first = install_vpi_data_pack_distribution(
        config_a,
        local_mirror_root=mirror_root,
    )
    config_b = VpiDataPackageConfig(
        descriptor_path=descriptor_path,
        install_dir=tmp_path / "installed-b",
        cache_dir=cache_dir,
        base_uri=None,
        package_version=descriptor.package_version,
    )
    second = install_vpi_data_pack_distribution(
        config_b,
        local_mirror_root=mirror_root,
    )
    assert first.install_report.files_downloaded >= 1
    assert second.install_report.files_downloaded == 0
    assert second.install_report.files_reused_from_cache >= 1


def test_corrupted_download_fails_closed(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, _, _ = _build_descriptor(tmp_path, monkeypatch)
    mirror_root = tmp_path / "mirror"
    _mirror_artifact(artifact_root, mirror_root)
    target = mirror_root / descriptor.files[0].relative_path
    target.write_bytes(b"corrupted")
    installer = DataPackageInstaller()
    with pytest.raises(DataPackageIntegrityError):
        installer.install(
            DataPackageInstallRequest(
                descriptor=descriptor,
                install_root=tmp_path / "installed",
                cache=DataPackageCache(tmp_path / "cache"),
                transport=LocalFileDataPackageTransport(),
                base_uri=mirror_root.as_uri() + "/",
            )
        )


def test_partial_install_not_published(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, _, _ = _build_descriptor(tmp_path, monkeypatch)
    mirror_root = tmp_path / "mirror"
    _mirror_artifact(artifact_root, mirror_root)
    (mirror_root / descriptor.files[0].relative_path).unlink()
    installer = DataPackageInstaller()
    with pytest.raises(Exception):
        installer.install(
            DataPackageInstallRequest(
                descriptor=descriptor,
                install_root=tmp_path / "installed",
                cache=DataPackageCache(tmp_path / "cache"),
                transport=LocalFileDataPackageTransport(),
                base_uri=mirror_root.as_uri() + "/",
            )
        )
    assert not (tmp_path / "installed" / descriptor.files[-1].relative_path).is_file()


def test_failed_reinstall_preserves_existing_installation(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, _, descriptor_path = _build_descriptor(tmp_path, monkeypatch)
    mirror_root = tmp_path / "mirror"
    _mirror_artifact(artifact_root, mirror_root)
    install_dir = tmp_path / "installed"
    cache_dir = tmp_path / "cache"
    config = VpiDataPackageConfig(
        descriptor_path=descriptor_path,
        install_dir=install_dir,
        cache_dir=cache_dir,
        base_uri=None,
        package_version=descriptor.package_version,
    )
    install_vpi_data_pack_distribution(
        config,
        local_mirror_root=mirror_root,
    )
    target = descriptor.files[0]
    before_manifest = (install_dir / "manifest" / "manifest.json").read_bytes()
    (install_dir / target.relative_path).unlink(missing_ok=True)
    cache = DataPackageCache(cache_dir)
    cache_path = cache.object_path(target.sha256)
    if cache_path.is_file():
        cache_path.unlink()
    (mirror_root / target.relative_path).write_bytes(b"broken")
    with pytest.raises(DataPackageIntegrityError):
        install_vpi_data_pack_distribution(
            config,
            local_mirror_root=mirror_root,
        )
    assert (install_dir / "manifest" / "manifest.json").read_bytes() == before_manifest


def test_http_resume_semantics_reused(tmp_path: Path) -> None:
    payload = b"0123456789"
    digest = hashlib.sha256(payload).hexdigest()
    mirror_root = tmp_path / "mirror"
    mirror_root.mkdir()
    (mirror_root / "resume.bin").write_bytes(payload)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            rel = parsed.path.lstrip("/")
            file_path = mirror_root / rel
            if not file_path.is_file():
                self.send_error(404)
                return
            data = file_path.read_bytes()
            if self.headers.get("Range", "").startswith("bytes="):
                start = int(self.headers["Range"].split("=")[1].split("-")[0])
                chunk = data[start:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
            else:
                chunk = data
                self.send_response(200)
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_uri = f"http://127.0.0.1:{server.server_address[1]}/"
    cache = DataPackageCache(tmp_path / "cache")
    partial = cache.partial_path(digest)
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(payload[:5])
    transport = HttpDataPackageTransport(max_retries=1)
    transport.download_file(f"{base_uri}resume.bin", partial, resume_from_byte=5)
    server.shutdown()
    assert hashlib.sha256(partial.read_bytes()).hexdigest() == digest


def test_distribution_modules_have_no_database_or_model_imports() -> None:
    forbidden_fragments = (
        "qdrant",
        "postgresql",
        "mysql",
        "pgvector",
        "torch",
        "sentence_transformers",
        "transformers",
    )
    violations: list[str] = []
    for module_path in sorted(_DATA_PACKAGE_ROOT.rglob("*.py")):
        for imported in _module_imports(module_path):
            if any(fragment in imported for fragment in forbidden_fragments):
                violations.append(f"{module_path.name} -> {imported}")
    assert violations == []


def test_redistribution_review_required_not_public_approved() -> None:
    assert not is_publication_approved(PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED)
    with pytest.raises(VpiDataPackageDescriptorBuildError):
        assert_publication_permitted(PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED)


def test_local_install_does_not_require_public_url(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, fixture, descriptor_path = _build_descriptor(tmp_path, monkeypatch)
    mirror_root = tmp_path / "mirror"
    _mirror_artifact(artifact_root, mirror_root)
    config = VpiDataPackageConfig(
        descriptor_path=descriptor_path,
        install_dir=tmp_path / "installed",
        cache_dir=tmp_path / "cache",
        base_uri=None,
        package_version=descriptor.package_version,
    )
    result = install_vpi_data_pack_distribution(
        config,
        local_mirror_root=mirror_root,
    )
    assert config.base_uri is None
    assert result.install_report.install_location.is_dir()


def test_content_policy_represents_required_paths(tmp_path: Path, monkeypatch) -> None:
    descriptor, artifact_root, fixture, _ = _build_descriptor(tmp_path, monkeypatch)
    files = collect_distributable_files(artifact_root)
    roles = {entry.role.value for entry in files}
    assert "RELATIONAL_SHARD" in roles
    assert "EMBEDDING_SHARD" in roles
    assert "MANIFEST" in roles
    assert "CHECKSUMS" in roles
    assert "PROOF_REPORT" in roles
    summary = summarize_descriptor(descriptor)
    assert summary.relational_file_count == fixture.expectations.shard_count
    assert summary.embedding_file_count == fixture.expectations.shard_count


def test_build_state_excluded_from_distribution(tmp_path: Path, monkeypatch) -> None:
    fixture = build_valid_validation_fixture(tmp_path, monkeypatch)
    paths = resolve_data_pack_paths(fixture.artifact_root)
    paths.build_state_file.parent.mkdir(parents=True, exist_ok=True)
    paths.build_state_file.write_text('{"status":"READY"}', encoding="utf-8")
    files = collect_distributable_files(fixture.artifact_root)
    assert all(entry.relative_path != "state/build-state.json" for entry in files)
    assert BUILD_STATE_EXCLUSION_REASON
    assert DISTRIBUTION_STRATEGY == "FILE_PER_SHARD"


def test_validation_gate_loader(tmp_path: Path, monkeypatch) -> None:
    _, report_path = _validated_fixture(tmp_path, monkeypatch)
    gate = load_descriptor_validation_gate(report_path)
    assert gate.verdict is DataPackValidationVerdict.PASS
    assert gate.finalized_artifact_valid is True


def test_forbidden_patterns_absent_in_distribution_modules() -> None:
    forbidden_fragments = (
        "dict[str, Any]",
        "dict[str, object]",
        "getattr",
        "setattr",
        "hasattr",
        "inspect",
    )
    violations: list[str] = []
    for module_path in sorted(_DATA_PACKAGE_ROOT.rglob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            if fragment in source:
                violations.append(f"{module_path.name} contains {fragment}")
    assert violations == []
