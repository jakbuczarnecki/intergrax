"""P7-00 throw-away adversarial probes for the PDC-X D6 / contracts paths (auditor-owned, read-only).

Run: PYTHONPATH=<checkout> python p7_00_probes_d6_security.py
Writes JSON results to stdout. No product, test or ticket file is modified.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

results: dict[str, object] = {}


def probe(name):
    def deco(fn):
        try:
            results[name] = fn()
        except Exception as exc:  # noqa: BLE001 - probe harness
            results[name] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        return fn

    return deco


# --------------------------------------------------------------------------- signing / strict config
@probe("S1_signing_uses_os_environ_when_environ_none")
def _():
    from intergrax.applications._shared.critical_action_signing_wiring import resolve_critical_action_signing_wiring
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
    from intergrax.core.security.security_configuration import SecurityConfigurationError

    env = ApplicationEnvironmentProfile.product_defaults()
    name = env.identity_profile.critical_action_signing_secret_env
    out = {"secret_env_name": name, "enabled_default": env.identity_profile.critical_action_signing_enabled}
    os.environ.pop(name, None)
    try:
        resolve_critical_action_signing_wiring(env)
        out["missing"] = "NO_RAISE"
    except SecurityConfigurationError as exc:
        out["missing"] = exc.reason_code
    os.environ[name] = "x" * 5
    try:
        out["short_secret_accepted"] = resolve_critical_action_signing_wiring(env).enabled
    finally:
        os.environ.pop(name, None)
    return out


@probe("S2_signing_disabled_flag_silently_disables_on_product")
def _():
    from intergrax.applications._shared.critical_action_signing_wiring import resolve_critical_action_signing_wiring
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.product_defaults()
    env.identity_profile = env.identity_profile.model_copy(update={"critical_action_signing_enabled": False})
    w = resolve_critical_action_signing_wiring(env, environ={})
    return {"enabled": w.enabled}


@probe("S3_strict_config_validator_collects_all_issues")
def _():
    from intergrax.applications._shared.security_assembly_resolver import validate_strict_security_configuration
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.product_defaults()
    issues = validate_strict_security_configuration(env, environ={})
    return [i.reason_code for i in issues]


@probe("S4_poisoning_default_matrix")
def _():
    from intergrax.core.security.security_configuration import retrieval_poisoning_defense_enabled_or_default as f

    class P:
        def __init__(self, v):
            self.retrieval_poisoning_defense_enabled = v

    return {repr(v): f(P(v)) for v in (False, True, None, 0, "false", "", [])} | {"absent_profile": f(None), "no_attr": f(object())}


# --------------------------------------------------------------------------- raw secret gate evasion
@probe("G1_raw_secret_gate_evasion_matrix")
def _():
    from intergrax.core.security.raw_secret_gate import scan_source_for_raw_secrets

    cases = {
        "direct_env_fallback": 'import os\nk = os.environ.get("API_KEY", "abcdef123456")\n',
        "getenv_fallback": 'import os\nk = os.getenv("SIGNING_SECRET", "abcdef123456")\n',
        "subscript_default_or": 'import os\nk = os.environ.get("API_KEY") or "abcdef123456"\n',
        "concat_literal": 'password = "abc" + "def123456789xyz"\n',
        "fstring_literal": 'password = f"abcdef{1}123456789xyz"\n',
        "dict_literal_key": 'cfg = {"api_key": "aB3dE5gH7jK9mN1pQ3sT5vX7zA9"}\n',
        "bytes_literal": 'secret = b"aB3dE5gH7jK9mN1pQ3sT5vX7zA9"\n',
        "plain_named_literal": 'api_key = "aB3dE5gH7jK9mN1pQ3sT5vX7zA9"\n',
        "sk_token_split": 'tok = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"\n',
        "sk_token_full": 'tok = "sk-A1b2C3d4E5f6G7h8I9j0K1l2M3n4"\n',
        "env_default_kw": 'import os\nk = os.environ.get("API_KEY", default="abcdef123456")\n',
        "attr_assignment": 'class C: pass\nc = C()\nc.password = "aB3dE5gH7jK9mN1pQ3sT5vX7zA9"\n',
        "dataclass_default": 'from dataclasses import dataclass\n@dataclass\nclass C:\n    api_key: str = "aB3dE5gH7jK9mN1pQ3sT5vX7zA9"\n',
    }
    return {
        k: [f.rule for f in scan_source_for_raw_secrets(src, path="probe.py")]
        for k, src in cases.items()
    }


# --------------------------------------------------------------------------- plugin provenance
@probe("PV1_attestation_for_other_distribution_vouches_for_spec")
def _():
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from intergrax.agent_distribution.ed25519_package_attestation_verifier import Ed25519PackageAttestationVerifier
    from intergrax.agent_distribution.identity import AgentPackageIdentity
    from intergrax.agent_distribution.package_attestation import (
        AgentPackageAttestationStatement,
        StaticPublisherVerificationKeyProvider,
    )
    from intergrax.agent_distribution.trust import AgentPackageTrustPolicy
    from intergrax.core.plugins.admission import PluginProvenanceAdmission, PluginProvenanceAttestation
    from intergrax.core.plugins.discovery import EP_TOOLS, EntryPointSpec

    key = Ed25519PrivateKey.generate()
    pub = key.public_key().public_bytes_raw()
    digest = "sha256:" + "ab" * 32

    def attest(dist):
        ident = AgentPackageIdentity(distribution_package_id=dist, package_version="1.0.0", package_digest=digest)
        st = AgentPackageAttestationStatement.from_package_identity(package_identity=ident, publisher_id="pub", key_id="k1")
        sig = base64.b64encode(key.sign(st.canonical_bytes())).decode()
        return PluginProvenanceAttestation(
            distribution=dist, package_version="1.0.0", package_digest=digest, publisher_id="pub",
            attestation_id="a1", key_id="k1", signature_b64=sig,
        )

    adm = PluginProvenanceAdmission(
        trust_policy=AgentPackageTrustPolicy(),
        attestation_verifier=Ed25519PackageAttestationVerifier(key_provider=StaticPublisherVerificationKeyProvider({("pub", "k1"): pub})),
        attestations={"victim-plugin": attest("other-signed-plugin")},  # key != attestation.distribution
        installed_version_resolver=lambda _n: "1.0.0",
        clock=lambda: datetime(2026, 11, 13, tzinfo=UTC),
    )
    spec = EntryPointSpec(name="victim", group=EP_TOOLS, value="victim_plugin:P", distribution="victim-plugin")
    rej = adm.evaluate(spec)
    return {"rejection": None if rej is None else rej.reason_code.value}


@probe("PV2_digest_of_installed_files_is_never_consulted")
def _():
    import inspect

    from intergrax.core.plugins import admission

    src = inspect.getsource(admission.PluginProvenanceAdmission.evaluate)
    return {"mentions_hash_or_files": any(t in src for t in ("hashlib", "sha256(", "read_bytes", "RECORD", "files(")),
            "installed_check": "installed_version_resolver" in src}


# --------------------------------------------------------------------------- audit trail
@probe("A1_evidence_plane_audit_list_truncates_at_1000")
def _():
    from intergrax.runtime.events.store import resolve_runtime_event_persistence
    from intergrax.runtime.security.security_audit_trail import EvidencePlaneSecurityAuditTrail

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        port = resolve_runtime_event_persistence(db_path=Path(d) / "ev.sqlite3")
        trail = EvidencePlaneSecurityAuditTrail(port)
        ids = [trail.append(tenant_id="t1", action="a", actor_id="u", resource="r").entry_id for _ in range(1005)]
        listed = trail.list_entries(tenant_id="t1")
        listed_ids = [e.entry_id for e in listed]
        return {"appended": len(ids), "listed": len(listed), "last_appended_listed": ids[-1] in listed_ids,
                "first_appended_listed": ids[0] in listed_ids}


@probe("A2_audit_metadata_secret_value_under_benign_key")
def _():
    from intergrax.runtime.events.store import resolve_runtime_event_persistence
    from intergrax.runtime.security.security_audit_trail import EvidencePlaneSecurityAuditTrail

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        port = resolve_runtime_event_persistence(db_path=Path(d) / "ev.sqlite3")
        trail = EvidencePlaneSecurityAuditTrail(port)
        out = {}
        for label, meta in {
            "secret_key_name": {"password": "x"},
            "nested_secret_key_name": {"a": {"api_key": "x"}},
            "secret_value_benign_key": {"note": "sk-A1b2C3d4E5f6G7h8I9j0K1l2M3n4"},
        }.items():
            try:
                trail.append(tenant_id="t1", action="a", actor_id="u", resource="r", metadata=meta)
                out[label] = "accepted"
            except Exception as exc:  # noqa: BLE001
                out[label] = f"rejected:{type(exc).__name__}"
        return out


@probe("A3_audit_tenant_whitespace_normalisation_and_cross_tenant_list")
def _():
    from intergrax.runtime.events.store import resolve_runtime_event_persistence
    from intergrax.runtime.security.security_audit_trail import EvidencePlaneSecurityAuditTrail

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        port = resolve_runtime_event_persistence(db_path=Path(d) / "ev.sqlite3")
        trail = EvidencePlaneSecurityAuditTrail(port)
        trail.append(tenant_id="tenant-a", action="a", actor_id="u", resource="r")
        trail.append(tenant_id="tenant-b", action="b", actor_id="u", resource="r")
        return {"a": [e.action for e in trail.list_entries(tenant_id="tenant-a")],
                "b": [e.action for e in trail.list_entries(tenant_id=" tenant-b ")],
                "a_as_b": [e.action for e in trail.list_entries(tenant_id="tenant-A")]}


# --------------------------------------------------------------------------- contracts
@probe("C1_host_deployment_profile_accepts_unknown_keys")
def _():
    from intergrax.applications.contracts.environment_profile import HostDeploymentProfile

    p = HostDeploymentProfile(business_agents_deploy_enabld=True, lkw_daemon_bind_host="0.0.0.0")  # typo + wide bind
    out = {"typo_key_accepted": "business_agents_deploy_enabld" in (p.model_extra or {}),
           "typed_flag_value": p.business_agents_deploy_enabled}
    try:
        HostDeploymentProfile(nested={"a": 1})
        out["nested_value"] = "accepted"
    except Exception as exc:  # noqa: BLE001
        out["nested_value"] = f"rejected:{type(exc).__name__}"
    from intergrax.applications._shared.lkw_hybrid_daemon_wiring import LkwHybridDaemonDeployment

    d = LkwHybridDaemonDeployment.from_host_deployment(p)
    out["lkw_bind_host_wide_accepted"] = d.lkw_daemon_bind_host
    try:
        LkwHybridDaemonDeployment.from_host_deployment(HostDeploymentProfile(lkw_daemon_port=70000))
        out["lkw_port_70000"] = "accepted"
    except Exception as exc:  # noqa: BLE001
        out["lkw_port_70000"] = f"rejected:{type(exc).__name__}"
    return out


@probe("C2_task_artifact_sha256_validation")
def _():
    from intergrax.contracts.task_artifacts import ApplicationArtifactRef

    base = dict(artifact_id="a", kind="k", uri="u", task_id="t", owner_app_id="app", tenant_id="ten")
    out = {}
    for label, sha in {"empty": "", "lower": "a" * 64, "upper": "A" * 64, "prefixed": "sha256:" + "a" * 64, "short": "abc"}.items():
        try:
            ApplicationArtifactRef(**base, sha256=sha)
            out[label] = "accepted"
        except Exception as exc:  # noqa: BLE001
            out[label] = "rejected"
    return out


@probe("C3_run_artifact_bundle_empty_digest_durable_ref")
def _():
    from intergrax.contracts.task_artifacts import ApplicationArtifactRef, RunArtifactBundle

    ref = ApplicationArtifactRef(artifact_id="a", kind="k", uri="u", task_id="t", owner_app_id="app", tenant_id="ten", sha256="")
    try:
        RunArtifactBundle(task_id="t", application=[ref]).durable_refs()
        return "NO_RAISE"
    except Exception as exc:  # noqa: BLE001
        return type(exc).__name__


print(json.dumps(results, indent=2, default=str, sort_keys=True))
