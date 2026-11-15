# © Artur Czarnecki. All rights reserved.

"""PDC-D6 F-D6-05 (binding part) — provenance is bound to the attested distribution.

``PluginProvenanceAdmission`` finds the attestation by the operator's mapping key. An attestation of
another distribution stored under the key of the entry point's distribution used to admit that entry
point. The attested distribution must now equal the entry point's distribution. Digesting the installed
files is an excluded finding.
"""

from __future__ import annotations

import pytest
from intergrax.agent_distribution.ed25519_package_attestation_verifier import Ed25519PackageAttestationVerifier
from intergrax.agent_distribution.package_attestation import StaticPublisherVerificationKeyProvider
from intergrax.agent_distribution.trust import AgentPackageTrustPolicy
from intergrax.core.plugins.admission import (
    PluginAdmissionReasonCode,
    PluginProvenanceAdmission,
    PluginProvenanceAttestation,
)
from intergrax.core.plugins.discovery import EntryPointSpec
from tests.qualification.pdc_x.d6.test_d6_plugin_provenance import (
    _FIXED_NOW,
    _KEY_ID,
    _PUBLISHER,
    _VERSION,
    _Signer,
    _spec,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_VICTIM = "victim-plugin"
_OTHER = "other-signed-plugin"


def _admission(
    signer: _Signer,
    mapping: dict[str, PluginProvenanceAttestation],
    *,
    policy: AgentPackageTrustPolicy | None = None,
) -> PluginProvenanceAdmission:
    provider = StaticPublisherVerificationKeyProvider({(_PUBLISHER, _KEY_ID): signer.public_bytes})
    return PluginProvenanceAdmission(
        trust_policy=policy or AgentPackageTrustPolicy(),
        attestation_verifier=Ed25519PackageAttestationVerifier(key_provider=provider),
        attestations=mapping,
        installed_version_resolver=lambda _name: _VERSION,
        clock=lambda: _FIXED_NOW,
    )


def _reason(rejection: object) -> PluginAdmissionReasonCode | None:
    return getattr(rejection, "reason_code", None)


@pytest.mark.parametrize(
    "key",
    [_VICTIM, "Victim-Plugin", "victim_plugin", " victim-plugin "],
    ids=["exact", "case", "underscore", "padded"],
)
def test_an_attestation_of_another_distribution_does_not_admit_the_entry_point(key: str) -> None:
    signer = _Signer()
    other = signer.attest(distribution=_OTHER)

    rejection = _admission(signer, {key: other}).evaluate(_spec(distribution=_VICTIM))

    assert _reason(rejection) is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_the_rejection_does_not_echo_attestation_content() -> None:
    signer = _Signer()
    other = signer.attest(distribution=_OTHER)

    rejection = _admission(signer, {_VICTIM: other}).evaluate(_spec(distribution=_VICTIM))

    assert rejection is not None
    assert other.signature_b64 not in rejection.reason
    assert other.package_digest not in rejection.reason


@pytest.mark.parametrize(
    ("key", "distribution", "spec_distribution"),
    [
        (_VICTIM, _VICTIM, _VICTIM),
        ("Victim_Plugin", "victim-plugin", "victim.plugin"),
        ("victim-plugin", "VICTIM_PLUGIN", "victim-plugin"),
    ],
    ids=["same", "key-and-spec-differ-by-normalization", "attestation-differs-by-normalization"],
)
def test_a_matching_attestation_is_admitted_under_the_normalization_rules(
    key: str, distribution: str, spec_distribution: str
) -> None:
    signer = _Signer()
    attestation = signer.attest(distribution=distribution)

    assert _admission(signer, {key: attestation}).evaluate(_spec(distribution=spec_distribution)) is None


def test_an_entry_point_without_a_distribution_is_still_missing_provenance_in_the_strict_policy() -> None:
    signer = _Signer()
    mapping = {_VICTIM: signer.attest(distribution=_VICTIM)}

    rejection = _admission(signer, mapping).evaluate(_spec(distribution=None))

    assert _reason(rejection) is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING


def test_the_lab_policy_still_admits_unsigned_and_still_rejects_a_mismatched_attestation() -> None:
    signer = _Signer()
    lab = AgentPackageTrustPolicy(forbid_unsigned_or_unqualified=False)

    assert _admission(signer, {}, policy=lab).evaluate(_spec(distribution=_VICTIM)) is None
    rejection = _admission(signer, {_VICTIM: signer.attest(distribution=_OTHER)}, policy=lab).evaluate(
        _spec(distribution=_VICTIM)
    )
    assert _reason(rejection) is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID


def test_installed_version_and_signature_checks_keep_their_rejections() -> None:
    signer = _Signer()
    attacker = _Signer()
    spec: EntryPointSpec = _spec(distribution=_VICTIM)

    forged = _admission(signer, {_VICTIM: attacker.attest(distribution=_VICTIM)}).evaluate(spec)
    assert _reason(forged) is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID

    stale = PluginProvenanceAdmission(
        trust_policy=AgentPackageTrustPolicy(),
        attestation_verifier=Ed25519PackageAttestationVerifier(
            key_provider=StaticPublisherVerificationKeyProvider({(_PUBLISHER, _KEY_ID): signer.public_bytes})
        ),
        attestations={_VICTIM: signer.attest(distribution=_VICTIM)},
        installed_version_resolver=lambda _name: "0.0.1",
        clock=lambda: _FIXED_NOW,
    ).evaluate(spec)
    assert _reason(stale) is PluginAdmissionReasonCode.PLUGIN_PROVENANCE_INVALID
