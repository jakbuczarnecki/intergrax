# © Artur Czarnecki. All rights reserved.

"""Flat ↔ nested profile normalization (APP-EVOL-8 · ADR-APP-003)."""

from __future__ import annotations

from typing import Any

BUNDLE_ROOT_KEYS: frozenset[str] = frozenset(
    {
        "meta",
        "security",
        "capabilities",
        "cognition",
        "governance",
        "topology",
        "isolation",
        "extensions",
    }
)

FLAT_PROFILE_KEYS: frozenset[str] = frozenset(
    {
        "profile_id",
        "spec_version",
        "application_profile",
        "execution_mode",
        "features",
        "identity_profile",
        "security_profile",
        "guardrail_profile",
        "policy_rules",
        "compliance_profile",
        "organizational_policy",
        "integration_profile",
        "tool_profile",
        "skill_profile",
        "modality_profile",
        "llm_profile",
        "prompt_profile",
        "context_profile",
        "memory_profile",
        "tool_selection_mode",
        "tool_selection_top_k",
        "tool_invocation_mode",
        "max_parallel_tool_calls",
        "reasoning_profile",
        "orchestration_profile",
        "critic_profile",
        "adaptive_profile",
        "evaluation_profile",
        "codecraft_profile",
        "reliability_profile",
        "observability_profile",
        "cost_profile",
        "scaling_profile",
        "governance_profile",
        "capability_governance_profile",
        "agent_governance_profile",
        "integration_governance_profile",
        "host_deployment_profile",
        "execution_boundary_export_profile",
        "graph_spec",
        "shadow_workspace",
        "sandbox",
        "domain_policy_fragments",
    }
)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {}


def lift_flat_profile_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy flat profile JSON into nested bundle fields."""
    if not isinstance(data, dict):
        return data
    if not any(key in data for key in FLAT_PROFILE_KEYS):
        return data

    meta = _as_dict(data.get("meta"))
    security = _as_dict(data.get("security"))
    capabilities = _as_dict(data.get("capabilities"))
    cognition = _as_dict(data.get("cognition"))
    governance = _as_dict(data.get("governance"))
    topology = _as_dict(data.get("topology"))
    isolation = _as_dict(data.get("isolation"))
    extensions = _as_dict(data.get("extensions"))

    if "profile_id" in data:
        meta["profile_id"] = data["profile_id"]
    if "spec_version" in data:
        meta["spec_version"] = data["spec_version"]
    if "application_profile" in data:
        meta["application_profile"] = data["application_profile"]
    if "execution_mode" in data:
        meta["execution_mode"] = data["execution_mode"]
    if "features" in data:
        meta["features"] = data["features"]

    for flat_key, nested_key in (
        ("identity_profile", "identity"),
        ("security_profile", "application_security"),
        ("guardrail_profile", "guardrails"),
        ("policy_rules", "policy_rules"),
        ("compliance_profile", "compliance"),
        ("organizational_policy", "organizational_policy"),
    ):
        if flat_key in data:
            security[nested_key] = data[flat_key]

    for flat_key, nested_key in (
        ("integration_profile", "integrations"),
        ("tool_profile", "tools"),
        ("skill_profile", "skills"),
        ("llm_profile", "llm"),
        ("modality_profile", "modality"),
        ("prompt_profile", "prompt"),
        ("context_profile", "context"),
        ("memory_profile", "memory"),
    ):
        if flat_key in data:
            capabilities[nested_key] = data[flat_key]

    tool_selection = _as_dict(capabilities.get("tool_selection"))
    if "tool_selection_mode" in data:
        tool_selection["mode"] = data["tool_selection_mode"]
    if "tool_selection_top_k" in data:
        tool_selection["top_k"] = data["tool_selection_top_k"]
    if tool_selection:
        capabilities["tool_selection"] = tool_selection

    tool_invocation = _as_dict(capabilities.get("tool_invocation"))
    if "tool_invocation_mode" in data:
        tool_invocation["mode"] = data["tool_invocation_mode"]
    if "max_parallel_tool_calls" in data:
        tool_invocation["max_parallel"] = data["max_parallel_tool_calls"]
    if tool_invocation:
        capabilities["tool_invocation"] = tool_invocation

    for flat_key, nested_key in (
        ("reasoning_profile", "reasoning"),
        ("orchestration_profile", "orchestration"),
        ("critic_profile", "critic"),
        ("adaptive_profile", "adaptive"),
        ("evaluation_profile", "evaluation"),
        ("codecraft_profile", "codecraft"),
    ):
        if flat_key in data:
            cognition[nested_key] = data[flat_key]

    for flat_key, nested_key in (
        ("reliability_profile", "reliability"),
        ("observability_profile", "observability"),
        ("cost_profile", "cost"),
        ("scaling_profile", "scaling"),
        ("governance_profile", "platform"),
        ("capability_governance_profile", "capability"),
        ("agent_governance_profile", "agent"),
        ("integration_governance_profile", "integration_marketplace"),
        ("host_deployment_profile", "deployment"),
        ("execution_boundary_export_profile", "boundary_export"),
    ):
        if flat_key in data:
            governance[nested_key] = data[flat_key]

    if "graph_spec" in data:
        topology["graph_spec"] = data["graph_spec"]
    if "shadow_workspace" in data:
        isolation["shadow_workspace"] = data["shadow_workspace"]
    if "sandbox" in data:
        isolation["sandbox"] = data["sandbox"]
    if "domain_policy_fragments" in data:
        extensions["domain_policy_fragments"] = data["domain_policy_fragments"]

    lifted = {
        key: value
        for key, value in data.items()
        if key not in FLAT_PROFILE_KEYS and key not in BUNDLE_ROOT_KEYS
    }
    lifted["meta"] = meta
    lifted["security"] = security
    lifted["capabilities"] = capabilities
    lifted["cognition"] = cognition
    lifted["governance"] = governance
    lifted["topology"] = topology
    lifted["isolation"] = isolation
    lifted["extensions"] = extensions
    return lifted


def flatten_profile_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Expand nested bundles to legacy flat wire shape (``spec_version`` 1.x)."""
    if not isinstance(data, dict):
        return data
    if not any(key in data for key in BUNDLE_ROOT_KEYS):
        return data

    meta = _as_dict(data.get("meta"))
    security = _as_dict(data.get("security"))
    capabilities = _as_dict(data.get("capabilities"))
    cognition = _as_dict(data.get("cognition"))
    governance = _as_dict(data.get("governance"))
    topology = _as_dict(data.get("topology"))
    isolation = _as_dict(data.get("isolation"))
    extensions = _as_dict(data.get("extensions"))
    tool_selection = _as_dict(capabilities.get("tool_selection"))
    tool_invocation = _as_dict(capabilities.get("tool_invocation"))

    flat: dict[str, Any] = {
        key: value
        for key, value in data.items()
        if key not in BUNDLE_ROOT_KEYS
    }
    flat.update(
        {
            "profile_id": meta.get("profile_id", "default"),
            "spec_version": meta.get("spec_version", "1.0.0"),
            "application_profile": meta.get("application_profile"),
            "execution_mode": meta.get("execution_mode"),
            "features": meta.get("features"),
            "identity_profile": security.get("identity"),
            "security_profile": security.get("application_security"),
            "guardrail_profile": security.get("guardrails"),
            "policy_rules": security.get("policy_rules"),
            "compliance_profile": security.get("compliance"),
            "organizational_policy": security.get("organizational_policy"),
            "integration_profile": capabilities.get("integrations"),
            "tool_profile": capabilities.get("tools"),
            "skill_profile": capabilities.get("skills"),
            "llm_profile": capabilities.get("llm"),
            "modality_profile": capabilities.get("modality"),
            "prompt_profile": capabilities.get("prompt"),
            "context_profile": capabilities.get("context"),
            "memory_profile": capabilities.get("memory"),
            "tool_selection_mode": tool_selection.get("mode", "static"),
            "tool_selection_top_k": tool_selection.get("top_k", 20),
            "tool_invocation_mode": tool_invocation.get("mode", "single_pass"),
            "max_parallel_tool_calls": tool_invocation.get("max_parallel", 8),
            "reasoning_profile": cognition.get("reasoning"),
            "orchestration_profile": cognition.get("orchestration"),
            "critic_profile": cognition.get("critic"),
            "adaptive_profile": cognition.get("adaptive"),
            "evaluation_profile": cognition.get("evaluation"),
            "codecraft_profile": cognition.get("codecraft"),
            "reliability_profile": governance.get("reliability"),
            "observability_profile": governance.get("observability"),
            "cost_profile": governance.get("cost"),
            "scaling_profile": governance.get("scaling"),
            "governance_profile": governance.get("platform"),
            "capability_governance_profile": governance.get("capability"),
            "agent_governance_profile": governance.get("agent"),
            "integration_governance_profile": governance.get("integration_marketplace"),
            "host_deployment_profile": governance.get("deployment"),
            "execution_boundary_export_profile": governance.get("boundary_export"),
            "graph_spec": topology.get("graph_spec"),
            "shadow_workspace": isolation.get("shadow_workspace"),
            "sandbox": isolation.get("sandbox"),
            "domain_policy_fragments": extensions.get("domain_policy_fragments"),
        }
    )
    return {key: value for key, value in flat.items() if value is not None}


def _strip_null_nodes(value: Any) -> Any:
    """Remove ``None`` leaves and empty dicts for stable bundle digests."""
    if isinstance(value, dict):
        cleaned = {
            key: _strip_null_nodes(item)
            for key, item in value.items()
            if item is not None
        }
        return {key: item for key, item in cleaned.items() if item != {}}
    if isinstance(value, list):
        return [_strip_null_nodes(item) for item in value if item is not None]
    return value


def bundle_normalized_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Canonical nested payload for stable digests (flat ≡ nested semantically)."""
    nested = lift_flat_profile_dict(data)
    payload = {key: nested[key] for key in sorted(nested) if key in BUNDLE_ROOT_KEYS}
    return _strip_null_nodes(payload)
