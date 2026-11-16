"""IA-D6 runtime mutation plugin (auditor-owned). Applies ONE in-memory mutation chosen by IA_MUT.

No file in the repository is edited; the mutation lives only in the pytest process.
"""
from __future__ import annotations

import os

MUT = os.environ.get("IA_MUT", "")


def _m1_signing_fallback():
    from intergrax.applications._shared import critical_action_signing_wiring as m

    orig = m.resolve_critical_action_signing_wiring

    def mutated(env, *, environ=None, lab_opt_in=None):
        try:
            return orig(env, environ=environ, lab_opt_in=lab_opt_in)
        except m.SecurityConfigurationError as exc:
            if exc.reason_code == m.SIGNING_SECRET_MISSING:
                return m._bootstrap(env, "harness-dev" + "-signing-key")
            raise

    m.resolve_critical_action_signing_wiring = mutated


def _m2_poison_default_off():
    from intergrax.core.security import security_configuration as m

    orig = m.retrieval_poisoning_defense_enabled_or_default

    def mutated(profile):
        if profile is None:
            return False
        return orig(profile)

    m.retrieval_poisoning_defense_enabled_or_default = mutated


def _m3_tool_guard_noop():
    from intergrax.applications._shared import application_security_wiring as m

    m.assert_tool_input_safe = lambda value: None


def _m4_provenance_allow_all():
    from intergrax.core.plugins import admission as m

    m.PluginProvenanceAdmission.evaluate = lambda self, spec: None


def _m5_demo_seed_ignore_strict():
    from intergrax.runtime.attestation import host_attestation as h
    from intergrax.runtime.execution_evidence import attestor as m

    m.resolve_demo_attestation_seed = lambda *, strict_profile: h.POC_ATTESTATION_DEMO_SIGNING_SEED


def _m6_broker_scope_noop():
    from intergrax.integrations.credentials import broker as m

    m.assert_grant_matches_scope = lambda grant, scope: None


def _m7_delegation_no_overreach():
    from intergrax.contracts import delegation_authority as m

    orig = m.mint_effective_delegation_authority

    def mutated(*, parent, requested_permission_scopes):
        if not parent.unrestricted and not parent.is_unknown:
            return m.EffectiveDelegationAuthority(
                requested_permission_scopes=tuple(requested_permission_scopes),
                parent_effective_scopes=parent.permission_scopes,
                parent_unrestricted=False,
                effective_permission_scopes=tuple(requested_permission_scopes),
            )
        return orig(parent=parent, requested_permission_scopes=requested_permission_scopes)

    m.mint_effective_delegation_authority = mutated


def _m8_secret_gate_blind():
    from intergrax.core.security import raw_secret_gate as m

    m.scan_source_for_raw_secrets = lambda source, *, path: ()


def _m9_audit_no_secret_validation():
    from intergrax.runtime.security import security_audit_trail as m

    m.validate_secret_safe_value = lambda *a, **k: None


def _m10_audit_not_durable():
    # evidence-plane trail forgets everything on a fresh instance (volatile): append is a no-op write
    from intergrax.runtime.security import security_audit_trail as m

    class _Null:
        def append(self, event, *, tenant_id):
            return None

        def list_for_task(self, *a, **k):
            return []

    orig_init = m.EvidencePlaneSecurityAuditTrail.__init__

    def init(self, evidence_persistence):
        orig_init(self, _Null())

    m.EvidencePlaneSecurityAuditTrail.__init__ = init


def _m11_strict_validator_clean():
    from intergrax.applications._shared import security_assembly_resolver as m

    m.validate_strict_security_configuration = lambda env, **kw: ()


def _m12_host_seed_always_lab():
    from intergrax.applications._shared import attestation_runtime_bridge as m

    m._is_strict_profile = lambda profile: False


_TABLE = {
    "M1": _m1_signing_fallback,
    "M2": _m2_poison_default_off,
    "M3": _m3_tool_guard_noop,
    "M4": _m4_provenance_allow_all,
    "M5": _m5_demo_seed_ignore_strict,
    "M6": _m6_broker_scope_noop,
    "M7": _m7_delegation_no_overreach,
    "M8": _m8_secret_gate_blind,
    "M9": _m9_audit_no_secret_validation,
    "M10": _m10_audit_not_durable,
    "M11": _m11_strict_validator_clean,
    "M12": _m12_host_seed_always_lab,
}

if MUT:
    _TABLE[MUT]()
