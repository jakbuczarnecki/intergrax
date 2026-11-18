# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-B Part D: installed-file digest check for plugins (NT-RW-SUP-06 = NT-PLG-05-1, EX-D6-03).

The fixture distribution is written to a temporary site directory; the publisher key is generated at
run time (no real key material). In-process fixture only.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from intergrax.agent_distribution.ed25519_package_attestation_verifier import Ed25519PackageAttestationVerifier
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationStatement,
    StaticPublisherVerificationKeyProvider,
)
from intergrax.agent_distribution.trust import AgentPackageTrustPolicy
from intergrax.core.plugins import admission as admission_module
from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginProvenanceAdmission,
    PluginProvenanceAttestation,
)
from intergrax.core.plugins.discovery import EP_TOOLS, EntryPointSpec

pytestmark = [pytest.mark.unit]

_DIST = "acme-tool-plugin"
_VERSION = "1.2.3"
_PUBLISHER = "acme-publisher"
_KEY_ID = "acme-key-1"
_NOW = datetime(2026, 11, 18, 9, 0, tzinfo=UTC)
_MODULE = "acme_tool_plugin/__init__.py"
_EXTRA = "acme_tool_plugin/tools.py"
_DIST_INFO = "acme_tool_plugin-1.2.3.dist-info"


def _record_hash(content: bytes) -> str:
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")


def _write_distribution(site: Path, *, record_files: tuple[str, ...] = (_MODULE, _EXTRA)) -> None:
    files = {
        _MODULE: b"VALUE = 1\n",
        _EXTRA: b"def tool() -> int:\n    return 1\n",
        f"{_DIST_INFO}/METADATA": f"Metadata-Version: 2.1\nName: {_DIST}\nVersion: {_VERSION}\n".encode(),
    }
    for relative, content in files.items():
        target = site / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    listed = (*record_files, f"{_DIST_INFO}/METADATA")
    lines = [f"{relative},{_record_hash(files[relative])},{len(files[relative])}" for relative in listed]
    lines.append(f"{_DIST_INFO}/RECORD,,")
    (site / _DIST_INFO / "RECORD").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tree_digest_fn() -> Callable[..., str | None]:
    digest_fn = getattr(admission_module, "installed_tree_digest", None)
    assert digest_fn is not None, "no installed-tree digest function exists (EX-D6-03)"
    return digest_fn


def _resolver(site: Path) -> Callable[[str], str | None]:
    digest_fn = _tree_digest_fn()
    return lambda name: digest_fn(name, search_path=[str(site)])


def _attest(private_key: Ed25519PrivateKey, digest: str) -> PluginProvenanceAttestation:
    identity = AgentPackageIdentity(distribution_package_id=_DIST, package_version=_VERSION, package_digest=digest)
    statement = AgentPackageAttestationStatement.from_package_identity(
        package_identity=identity, publisher_id=_PUBLISHER, key_id=_KEY_ID
    )
    return PluginProvenanceAttestation(
        distribution=_DIST,
        package_version=_VERSION,
        package_digest=digest,
        publisher_id=_PUBLISHER,
        attestation_id="att-1",
        key_id=_KEY_ID,
        signature_b64=base64.b64encode(private_key.sign(statement.canonical_bytes())).decode(),
    )


def _admission(site: Path, attested_digest: str) -> PluginProvenanceAdmission:
    assert "installed_digest_resolver" in {f.name for f in dataclasses.fields(PluginProvenanceAdmission)}, (
        "PluginProvenanceAdmission cannot re-hash installed files (EX-D6-03)"
    )
    private_key = Ed25519PrivateKey.generate()
    provider = StaticPublisherVerificationKeyProvider(
        {(_PUBLISHER, _KEY_ID): private_key.public_key().public_bytes_raw()}
    )
    base = PluginProvenanceAdmission(
        trust_policy=AgentPackageTrustPolicy(),
        attestation_verifier=Ed25519PackageAttestationVerifier(key_provider=provider),
        attestations={_DIST: _attest(private_key, attested_digest)},
        installed_version_resolver=lambda _name: _VERSION,
        clock=lambda: _NOW,
    )
    return dataclasses.replace(base, installed_digest_resolver=_resolver(site))


def _spec() -> EntryPointSpec:
    return EntryPointSpec(name="acme", group=EP_TOOLS, value="acme_tool_plugin:Plugin", distribution=_DIST)


def _intact_digest(site: Path) -> str:
    digest = _tree_digest_fn()(_DIST, search_path=[str(site)])
    assert isinstance(digest, str) and digest.startswith("sha256:")
    return digest


def test_sup_06a_altered_installed_file_refused(tmp_path: Path) -> None:
    _write_distribution(tmp_path)
    attested = _intact_digest(tmp_path)
    (tmp_path / _EXTRA).write_bytes(b"def tool() -> int:\n    return 2\n")
    rejection = _admission(tmp_path, attested).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID
    assert rejection.fail_closed is True


def test_sup_06b_missing_file_refused(tmp_path: Path) -> None:
    _write_distribution(tmp_path)
    attested = _intact_digest(tmp_path)
    _write_distribution(tmp_path, record_files=(_MODULE,))  # RECORD entry of tools.py removed
    rejection = _admission(tmp_path, attested).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID

    _write_distribution(tmp_path)
    (tmp_path / _EXTRA).unlink()  # listed in RECORD, missing on disk
    rejection = _admission(tmp_path, attested).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_sup_06c_matching_digest_admitted(tmp_path: Path) -> None:
    _write_distribution(tmp_path)
    attested = _intact_digest(tmp_path)
    assert _admission(tmp_path, attested).evaluate(_spec()) is None
    assert _intact_digest(tmp_path) == attested


def test_sup_06d_unlisted_extension_module_refused(tmp_path: Path) -> None:
    """An unlisted ``.pyd``/``.so`` beside a listed module would take import precedence (review M-3)."""
    _write_distribution(tmp_path)
    attested = _intact_digest(tmp_path)
    (tmp_path / "acme_tool_plugin" / "tools.cp312-win_amd64.pyd").write_bytes(b"MZ-planted")
    rejection = _admission(tmp_path, attested).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_sup_06e_unlisted_sourceless_bytecode_refused(tmp_path: Path) -> None:
    """A sourceless ``.pyc`` outside ``__pycache__`` is importable; it is not in RECORD (review M-3)."""
    _write_distribution(tmp_path)
    attested = _intact_digest(tmp_path)
    (tmp_path / "acme_tool_plugin" / "extra.pyc").write_bytes(b"planted-bytecode")
    rejection = _admission(tmp_path, attested).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_sup_06f_bytecode_cache_is_outside_the_digest(tmp_path: Path) -> None:
    """Documented limit: ``__pycache__`` files are not covered (install-time artefacts, plan Part D)."""
    _write_distribution(tmp_path)
    attested = _intact_digest(tmp_path)
    cache = tmp_path / "acme_tool_plugin" / "__pycache__"
    cache.mkdir()
    (cache / "tools.cpython-312.pyc").write_bytes(b"cache")
    assert _admission(tmp_path, attested).evaluate(_spec()) is None
