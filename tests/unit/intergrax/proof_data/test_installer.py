"""Unit tests for proof data package installer, cache, and transport."""

from __future__ import annotations

import hashlib
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
    load_proof_data_package_descriptor,
)


def _fixture_root() -> Path:
    return (
        Path(__file__).resolve().parents[4]
        / "platform_proofs"
        / "scenarios"
        / "verified_product_identification"
        / "data_package"
        / "fixtures"
        / "tiny_v1"
    )


def test_local_mirror_install(tmp_path: Path) -> None:
    fixture = _fixture_root()
    descriptor = load_proof_data_package_descriptor(fixture / "package.json")
    cache = DataPackageCache(tmp_path / "cache")
    install_root = tmp_path / "install"
    transport = LocalFileDataPackageTransport()
    installer = DataPackageInstaller()
    report = installer.install(
        DataPackageInstallRequest(
            descriptor=descriptor,
            install_root=install_root,
            cache=cache,
            transport=transport,
            base_uri=fixture.as_uri() + "/",
        )
    )
    assert report.files_total == len(descriptor.files)
    assert report.verification_passed is True
    assert (install_root / "dataset" / "selected_offers.parquet").is_file()


def test_cache_hit_on_second_install(tmp_path: Path) -> None:
    fixture = _fixture_root()
    descriptor = load_proof_data_package_descriptor(fixture / "package.json")
    cache = DataPackageCache(tmp_path / "cache")
    transport = LocalFileDataPackageTransport()
    installer = DataPackageInstaller()
    first = installer.install(
        DataPackageInstallRequest(
            descriptor=descriptor,
            install_root=tmp_path / "install1",
            cache=cache,
            transport=transport,
            base_uri=fixture.as_uri() + "/",
        )
    )
    second = installer.install(
        DataPackageInstallRequest(
            descriptor=descriptor,
            install_root=tmp_path / "install2",
            cache=cache,
            transport=transport,
            base_uri=fixture.as_uri() + "/",
        )
    )
    assert first.files_downloaded >= 1
    assert second.files_downloaded == 0
    assert second.files_reused_from_cache >= 1


def test_checksum_mismatch_fails_closed(tmp_path: Path) -> None:
    fixture = _fixture_root()
    descriptor = load_proof_data_package_descriptor(fixture / "package.json")
    cache = DataPackageCache(tmp_path / "cache")
    bad_file = descriptor.files[0]
    cache_path = cache.object_path(bad_file.sha256)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"corrupt")
    lookup = cache.lookup(bad_file.sha256, expected_size_bytes=bad_file.size_bytes)
    assert lookup.hit is False
    assert not cache_path.is_file()


class _RangeFileHandler(BaseHTTPRequestHandler):
    payload = b"0123456789"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        rel = parsed.path.lstrip("/")
        root = _fixture_root()
        file_path = root / rel
        if not file_path.is_file():
            self.send_response(404)
            self.end_headers()
            return
        data = file_path.read_bytes()
        if self.headers.get("Range", "").startswith("bytes="):
            start = int(self.headers["Range"].split("=")[1].split("-")[0])
            chunk = data[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data)-1}/{len(data)}")
        else:
            chunk = data
            self.send_response(200)
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        self.wfile.write(chunk)

    def log_message(self, format: str, *args: object) -> None:
        return


def test_http_range_resume(tmp_path: Path) -> None:
    fixture = _fixture_root()
    descriptor = load_proof_data_package_descriptor(fixture / "package.json")
    target = descriptor.files[0]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeFileHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_uri = f"http://127.0.0.1:{server.server_address[1]}/"
    cache = DataPackageCache(tmp_path / "cache")
    transport = HttpDataPackageTransport(max_retries=1)
    partial = cache.partial_path(target.sha256)
    partial.parent.mkdir(parents=True, exist_ok=True)
    source_path = fixture / target.relative_path
    data = source_path.read_bytes()
    partial.write_bytes(data[:5])
    transport.download_file(
        base_uri + target.relative_path,
        partial,
        resume_from_byte=5,
    )
    digest = hashlib.sha256(partial.read_bytes()).hexdigest()
    assert digest == target.sha256
    server.shutdown()
