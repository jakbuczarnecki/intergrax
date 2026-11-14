"""P7-00 addendum 1: throw-away probes for the six delta paths (auditor-owned, read-only)."""

from __future__ import annotations

import json
import warnings

warnings.filterwarnings("ignore")
results: dict[str, object] = {}


def probe(name):
    def deco(fn):
        try:
            results[name] = fn()
        except BaseException as exc:  # noqa: BLE001
            results[name] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        return fn

    return deco


@probe("Q1_memory_tenant_guard_is_a_noop_for_a_non_UserProfileManager_capability")
def _():
    from intergrax.memory.contracts.memory_control import MemoryControlAccessDenied
    from intergrax.memory.default_memory_control_plane import (
        DefaultMemoryControlPlane,
        UserProfileManagerMemoryCapability,
    )
    from intergrax.memory.user_profile_manager import UserProfileManager
    from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
    from intergrax.contracts.agent_run import RequestIdentity

    foreign = RequestIdentity(tenant_id="tenant-B", user_id="u")
    bound = DefaultMemoryControlPlane(
        user_profile=UserProfileManagerMemoryCapability(
            _manager=UserProfileManager(store=InMemoryUserProfileStore(), tenant_id="tenant-A")
        )
    )
    out = {}
    try:
        bound._assert_identity_tenant_is_bound_tenant(foreign)
        out["bound_plane_foreign_identity"] = "NO_RAISE"
    except MemoryControlAccessDenied:
        out["bound_plane_foreign_identity"] = "denied"

    class DuckCapability:  # protocol-shaped stand-in, not the shipped wrapper
        pass

    duck = DefaultMemoryControlPlane(user_profile=DuckCapability())  # type: ignore[arg-type]
    try:
        duck._assert_identity_tenant_is_bound_tenant(foreign)
        out["duck_capability_foreign_identity"] = "NO_RAISE"
    except MemoryControlAccessDenied:
        out["duck_capability_foreign_identity"] = "denied"
    try:
        bound._assert_identity_tenant_is_bound_tenant(RequestIdentity(tenant_id="tenant-A ", user_id="u"))
        out["whitespace_variant"] = "NO_RAISE"
    except MemoryControlAccessDenied:
        out["whitespace_variant"] = "denied"
    return out


@probe("Q2_required_features_on_an_adapter_without_any_declaration")
def _():
    from intergrax.applications._shared.llm_resolver import resolve_llm_adapter
    from intergrax.llm_adapters.contracts.capabilities import LLMCapabilityMismatchError

    class Bare:
        provider = "bare"
        model = "m"

    try:
        resolve_llm_adapter(None, Bare(), required_llm_features=("tools",))  # type: ignore[arg-type]
        return "ACCEPTED_WITHOUT_DECLARATION"
    except LLMCapabilityMismatchError:
        return "refused:LLMCapabilityMismatchError"


@probe("Q3_agent_contract_required_features_edge_cases")
def _():
    from intergrax.contracts.agent_contract_meta import AgentContract

    out = {}
    c = AgentContract(id="a", name="A", description="d", required_llm_features=["tools", "tools", "vision"])
    out["duplicates_kept"] = [str(x) for x in c.required_llm_features]
    out["json_roundtrip_equal"] = AgentContract.model_validate(c.model_dump(mode="json")).required_llm_features == c.required_llm_features
    try:
        AgentContract(id="a", name="A", description="d", required_llm_features=["TOOLS"])
        out["upper_case_name"] = "accepted"
    except Exception as exc:  # noqa: BLE001
        out["upper_case_name"] = f"rejected:{type(exc).__name__}"
    return out


@probe("Q4_llm_profile_digest_sensitivity")
def _():
    from intergrax.agent_distribution.runtime_revision import build_llm_profile_digest
    from intergrax.llm_adapters.registry.profile import LLMProfile

    base = LLMProfile(provider="p", model="m", options={"max_retries": 2})
    return {
        "same_profile_same_digest": build_llm_profile_digest(base) == build_llm_profile_digest(LLMProfile(provider="p", model="m", options={"max_retries": 2})),
        "model_change": build_llm_profile_digest(base) != build_llm_profile_digest(LLMProfile(provider="p", model="m2", options={"max_retries": 2})),
        "options_change": build_llm_profile_digest(base) != build_llm_profile_digest(LLMProfile(provider="p", model="m", options={"max_retries": 3})),
        "fallback_change": build_llm_profile_digest(base) != build_llm_profile_digest(LLMProfile(provider="p", model="m", options={"max_retries": 2}, fallback_profiles=(LLMProfile(provider="q", model="x"),))),
        "routing_hint_change": build_llm_profile_digest(base) != build_llm_profile_digest(LLMProfile(provider="p", model="m", options={"max_retries": 2}, routing_policy_hint="cheap")),
    }


@probe("Q5_production_callers_of_the_new_parameters")
def _():
    import subprocess

    def count(pattern):
        out = subprocess.run(["git", "grep", "-n", pattern, "HEAD", "--", "intergrax", "applications", "agents"], capture_output=True, text=True).stdout
        return [ln for ln in out.splitlines() if "def " not in ln and "llm_resolver.py" not in ln and "admin_service.py" not in ln and "agent_contract_meta.py" not in ln]

    return {"required_llm_features_callers": count("required_llm_features="), "llm_profile_source_callers": count("llm_profile_source=")}


print(json.dumps(results, indent=2, default=str, sort_keys=True))
