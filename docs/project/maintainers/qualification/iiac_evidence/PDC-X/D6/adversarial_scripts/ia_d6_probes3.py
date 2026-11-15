"""IA-D6 probes, part 3 (auditor-owned, read-only)."""
from __future__ import annotations

import json
import traceback

R: dict[str, object] = {}


def probe(name):
    def deco(fn):
        try:
            R[name] = fn()
        except Exception as e:  # noqa: BLE001
            R[name] = {"probe_error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-500:]}
        return fn

    return deco


@probe("ID1_request_identity_whitespace_tenant_not_normalised")
def _():
    from intergrax.contracts.agent_run import RequestIdentity
    from intergrax.runtime.security.tenant_scope import normalize_tenant_scope_id

    ri = RequestIdentity(tenant_id="  tenant-a  ")
    return {"stored": repr(ri.tenant_id), "scope_normalised": normalize_tenant_scope_id(ri.tenant_id)}


@probe("CR1_provider_exception_chain_keeps_secret_text")
def _():
    from intergrax.integrations.contracts.credential import CredentialRef, CredentialResolutionContext
    from intergrax.integrations.credentials.secrets_store_resolver import SecretsStoreCredentialResolver
    from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

    class Boom:
        def get_secret(self, path, *, version=None):
            raise RuntimeError("backend failed token=tv-SUPERSECRET-123")

    ref = CredentialRef.from_secret_path(provider_id="p", secret_path="secrets/t/p", tenant_id="t")
    ctx = CredentialResolutionContext(
        tenant_id="t",
        execution_id="e",
        operation="op",
        integration_id="p:i",
        target_scope=canonicalize_network_egress_allowlist(("https://a.example:443",)),
    )
    try:
        SecretsStoreCredentialResolver(Boom()).resolve(ref, context=ctx)
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception(exc))
        return {
            "exc_type": type(exc).__name__,
            "message_has_secret": "SUPERSECRET" in str(exc),
            "full_traceback_has_secret": "SUPERSECRET" in tb,
        }
    return "no exception"


@probe("PL1_default_defense_plugin_admission_is_structural_only")
def _():
    from intergrax.runtime.security.defense_plugin_loader import SecurityDefenseAdmissionPolicy

    p = SecurityDefenseAdmissionPolicy()
    return {"provenance": p.provenance, "on_load_failure": p.on_load_failure}


@probe("BS1_bootstrap_catalogs_signature_defaults")
def _():
    import inspect

    from intergrax.core.catalog_bootstrap import bootstrap_catalogs

    sig = inspect.signature(bootstrap_catalogs)
    return {k: repr(v.default) for k, v in sig.parameters.items() if k in ("require_plugin_provenance", "plugin_provenance", "discover_entry_points")}


@probe("ENV1_product_defaults_security_toggles")
def _():
    from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile

    env = ApplicationEnvironmentProfile.product_defaults()
    s, i = env.security_profile, env.identity_profile
    return {
        "tool_injection_defense_enabled": s.tool_injection_defense_enabled,
        "retrieval_poisoning_defense_enabled": s.retrieval_poisoning_defense_enabled,
        "immutable_audit_trail_enabled": s.immutable_audit_trail_enabled,
        "audit_trail_regions": list(s.audit_trail_regions),
        "critical_action_signing_enabled": i.critical_action_signing_enabled,
        "execution_mode": str(env.execution_mode),
    }


print(json.dumps(R, indent=1, default=str))
