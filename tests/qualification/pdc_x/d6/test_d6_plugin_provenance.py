# © Artur Czarnecki. All rights reserved.

"""PDC-D6-R016 / OD-PDC-SCOPE-07 = A: plugin entry points need publisher provenance."""

from __future__ import annotations

import ast
import base64
import inspect
from datetime import UTC, datetime

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from intergrax.agent_distribution.ed25519_package_attestation_verifier import (
    Ed25519PackageAttestationVerifier,
)
from intergrax.agent_distribution.identity import AgentPackageIdentity
from intergrax.agent_distribution.package_attestation import (
    AgentPackageAttestationStatement,
    StaticPublisherVerificationKeyProvider,
)
from intergrax.agent_distribution.trust import (
    AgentPackageTrustPolicy,
    AgentPackageTrustRevocationState,
)
from intergrax.core import catalog_bootstrap
from intergrax.core.plugins import admission as admission_module
from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginProvenanceAdmission,
    PluginProvenanceAttestation,
)
from intergrax.core.plugins.discovery import EP_SECURITY_DEFENSES, EP_TOOLS, EntryPointLoadResult, EntryPointSpec
from intergrax.core.security.security_configuration import (
    PLUGIN_TRUST_POLICY_MISSING,
    SecurityConfigurationError,
)
from intergrax.runtime.security import defense_plugin_loader

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_DIGEST = "sha256:" + "ab" * 32
_PUBLISHER = "acme-publisher"
_KEY_ID = "acme-key-1"
_DIST = "acme-tool-plugin"
_VERSION = "1.2.3"
_FIXED_NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


class _Signer:
    def __init__(self) -> None:
        self.private_key = Ed25519PrivateKey.generate()
        self.public_bytes = self.private_key.public_key().public_bytes_raw()

    def attest(
        self,
        *,
        distribution: str = _DIST,
        version: str = _VERSION,
        digest: str = _DIGEST,
        publisher: str = _PUBLISHER,
        key_id: str = _KEY_ID,
    ) -> PluginProvenanceAttestation:
        identity = AgentPackageIdentity(
            distribution_package_id=distribution,
            package_version=version,
            package_digest=digest,
        )
        statement = AgentPackageAttestationStatement.from_package_identity(
            package_identity=identity, publisher_id=publisher, key_id=key_id
        )
        signature = base64.b64encode(self.private_key.sign(statement.canonical_bytes())).decode()
        return PluginProvenanceAttestation(
            distribution=distribution,
            package_version=version,
            package_digest=digest,
            publisher_id=publisher,
            attestation_id="att-1",
            key_id=key_id,
            signature_b64=signature,
        )


def _admission(
    signer: _Signer,
    *attestations: PluginProvenanceAttestation,
    policy: AgentPackageTrustPolicy | None = None,
    revocation: AgentPackageTrustRevocationState | None = None,
    installed_version: str | None = _VERSION,
    trusted_key: bytes | None = None,
) -> PluginProvenanceAdmission:
    provider = StaticPublisherVerificationKeyProvider({(_PUBLISHER, _KEY_ID): trusted_key or signer.public_bytes})
    return PluginProvenanceAdmission(
        trust_policy=policy or AgentPackageTrustPolicy(),
        attestation_verifier=Ed25519PackageAttestationVerifier(key_provider=provider),
        attestations={item.distribution: item for item in attestations},
        revocation_state=revocation,
        installed_version_resolver=lambda _name: installed_version,
        clock=lambda: _FIXED_NOW,
    )


def _spec(distribution: str | None = _DIST, name: str = "acme", group: str = EP_TOOLS) -> EntryPointSpec:
    return EntryPointSpec(name=name, group=group, value="acme_plugin:Plugin", distribution=distribution)


def test_d6_r016_signed_plugin_is_admitted() -> None:
    signer = _Signer()
    assert _admission(signer, signer.attest()).evaluate(_spec()) is None


def test_d6_r016_unsigned_plugin_is_rejected_in_strict_profile() -> None:
    signer = _Signer()
    rejection = _admission(signer).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING
    assert rejection.fail_closed is True


def test_d6_r016_entry_point_without_distribution_is_rejected_in_strict_profile() -> None:
    signer = _Signer()
    rejection = _admission(signer, signer.attest()).evaluate(_spec(distribution=None))
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING


def test_d6_r016_lab_policy_admits_unsigned_but_never_an_invalid_signature() -> None:
    signer = _Signer()
    lab_policy = AgentPackageTrustPolicy(forbid_unsigned_or_unqualified=False)
    assert _admission(signer, policy=lab_policy).evaluate(_spec()) is None
    forged = _Signer().attest()  # signed by a key that is not pinned for the publisher
    rejection = _admission(signer, forged, policy=lab_policy).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_d6_r016_signature_by_an_unpinned_key_is_rejected() -> None:
    signer = _Signer()
    attacker = _Signer()
    rejection = _admission(signer, attacker.attest()).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_d6_r016_tampered_digest_is_rejected() -> None:
    signer = _Signer()
    genuine = signer.attest()
    tampered = PluginProvenanceAttestation(
        distribution=genuine.distribution,
        package_version=genuine.package_version,
        package_digest="sha256:" + "cd" * 32,
        publisher_id=genuine.publisher_id,
        attestation_id=genuine.attestation_id,
        key_id=genuine.key_id,
        signature_b64=genuine.signature_b64,
    )
    rejection = _admission(signer, tampered).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_d6_r016_installed_version_must_match_the_attested_version() -> None:
    signer = _Signer()
    for installed in ("9.9.9", None):
        rejection = _admission(signer, signer.attest(), installed_version=installed).evaluate(_spec())
        assert rejection is not None
        assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_d6_r016_trust_policy_denials_apply_to_plugins() -> None:
    signer = _Signer()
    attestation = signer.attest()
    cases = {
        "denied publisher": AgentPackageTrustPolicy(denied_publisher_ids=frozenset({_PUBLISHER})),
        "publisher not permitted": AgentPackageTrustPolicy(permitted_publisher_ids=frozenset({"someone-else"})),
        "denied digest": AgentPackageTrustPolicy(denied_package_digests=frozenset({_DIGEST})),
    }
    for label, policy in cases.items():
        rejection = _admission(signer, attestation, policy=policy).evaluate(_spec())
        assert rejection is not None, label
        assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_DENIED, label


def test_d6_r016_revoked_publisher_is_rejected() -> None:
    signer = _Signer()
    revocation = AgentPackageTrustRevocationState(revoked_publisher_ids=frozenset({_PUBLISHER}))
    rejection = _admission(signer, signer.attest(), revocation=revocation).evaluate(_spec())
    assert rejection is not None
    assert rejection.reason_code is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_DENIED


def test_d6_r016_admission_reuses_the_agent_package_trust_mechanism() -> None:
    """No second trust or signature authority: no crypto primitives in the admission module."""
    tree = ast.parse(inspect.getsource(admission_module))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(name.startswith(("cryptography", "hmac", "hashlib", "nacl")) for name in imported)
    assert "intergrax.agent_distribution.package_trust" in imported
    assert "intergrax.agent_distribution.package_attestation" in imported
    assert "AgentPackageTrustCoordinator" in inspect.getsource(admission_module)


def _patch_catalog_discovery(
    monkeypatch: pytest.MonkeyPatch,
    specs: tuple[EntryPointSpec, ...],
    loaded: list[str],
) -> None:
    monkeypatch.setattr(
        catalog_bootstrap,
        "iter_entry_point_specs",
        lambda group: specs if group == EP_TOOLS else (),
    )

    def _load(selected, *, on_load_failure="fail_fast"):
        results = []
        for spec in selected:
            loaded.append(spec.name)
            results.append(EntryPointLoadResult(spec=spec, target=type("NotAPlugin", (), {})))
        return results

    monkeypatch.setattr(catalog_bootstrap, "load_entry_point_targets_for_specs", _load)


def test_d6_r016_bootstrap_rejects_unsigned_plugin_before_it_is_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer = _Signer()
    loaded: list[str] = []
    unsigned = _spec(distribution="unsigned-plugin", name="unsigned")
    signed = _spec(name="signed")
    _patch_catalog_discovery(monkeypatch, (signed, unsigned), loaded)
    result = catalog_bootstrap.bootstrap_catalogs(
        register_shipped=False,
        discover_tool_entry_points=True,
        plugin_provenance=_admission(signer, signer.attest()),
        require_plugin_provenance=True,
    )
    report = result.tool_plugin_load_report
    assert loaded == ["signed"], "unsigned plugin target must never be loaded"
    rejected = {item.spec.name: item.reason_code for item in report.rejected}
    assert rejected["unsigned"] is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING
    # the signed plugin passed provenance and reached registration (not a plugin class here)
    assert rejected["signed"] is PluginAdmissionReasonCode.INVALID_TARGET_TYPE
    assert report.critical_bootstrap_acceptable is False


def test_d6_r016_strict_discovery_without_a_trust_policy_fails_closed() -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        catalog_bootstrap.bootstrap_catalogs(
            register_shipped=False,
            discover_entry_points=True,
            require_plugin_provenance=True,
        )
    assert excinfo.value.reason_code == PLUGIN_TRUST_POLICY_MISSING


def test_d6_r016_without_provenance_behaviour_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake(group, register, *, discover_entry_points, on_conflict):
        calls.append(group)
        return catalog_bootstrap.DomainPluginLoadReport.empty(group)

    monkeypatch.setattr(catalog_bootstrap, "register_plugins_with_report", _fake)
    catalog_bootstrap.bootstrap_catalogs(register_shipped=False, discover_entry_points=True)
    assert len(calls) == 3


def test_d6_r016_security_defense_loader_applies_provenance_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signer = _Signer()
    unsigned = _spec(distribution="unsigned-defense", name="unsigned-defense", group=EP_SECURITY_DEFENSES)
    seen: dict[str, object] = {}
    monkeypatch.setattr(defense_plugin_loader, "iter_entry_point_specs", lambda _group: (unsigned,))

    def _load(group, *, on_conflict, on_load_failure, skip_names=frozenset()):
        seen["skip_names"] = skip_names
        return []

    monkeypatch.setattr(defense_plugin_loader, "load_entry_point_targets", _load)
    report = defense_plugin_loader.load_security_defense_plugin_report(
        admission=defense_plugin_loader.SecurityDefenseAdmissionPolicy(provenance=_admission(signer))
    )
    assert seen["skip_names"] == frozenset({"unsigned-defense"})
    assert [item.reason_code for item in report.rejected] == [PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING]
