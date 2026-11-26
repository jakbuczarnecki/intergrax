# © Artur Czarnecki. All rights reserved.

"""ARCH-FREEZE package D — skill instruction and policy pinning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from intergrax.applications._shared.skill_bridge_wiring import (
    SkillBridgeCompositionError,
    merge_skill_policy_fragments,
)
from intergrax.contracts.execution_identity import mint_execution_id
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.runtime.policy.policy_bundle import DeclarativePolicyRuntime, RuntimePolicyBundle
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.runtime.policy.rules.provenance import PolicyBundleProvenance
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule, PolicyRuleAction
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.execution_binding import (
    InMemorySkillExecutionPinningStore,
    bind_resolved_skill_pack,
)
from intergrax.skills.instruction_policy_pins import resolve_pinned_prompt_instructions
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolver

pytestmark = pytest.mark.unit


def _prompt_catalog(tmp_path: Path, *, prompt_id: str, version: int, body: str) -> YamlPromptRegistry:
    prompt_dir = tmp_path / prompt_id
    prompt_dir.mkdir(parents=True)
    (prompt_dir / f"{version}.yaml").write_text(
        f"""
id: {prompt_id}
version: {version}
locales:
  en:
    content:
      system: "{body}"
      developer: null
      user_template: null
meta:
  model_family: generic
  output_schema_id: test.v1
  tags: [test]
""",
        encoding="utf-8",
    )
    (prompt_dir / "stable.yaml").write_text(f"stable: {version}\n", encoding="utf-8")
    registry = YamlPromptRegistry(catalog_dir=tmp_path)
    registry.load_all()
    return registry


def _policy_bundle_with_fragment(
    fragment_id: str,
    payload: dict,
    *,
    host_rules: tuple[DeclarativePolicyRule, ...] = (),
) -> RuntimePolicyBundle:
    registry = PolicyRuleRegistry()
    provenance = PolicyBundleProvenance(
        source_kind="inline",
        rules_digest_sha256="test",
        handler_provenance=(),
        rejected_handler_ids=(),
    )
    runtime = DeclarativePolicyRuntime(
        registry=registry,
        rules=host_rules,
        load_report=DomainPluginLoadReport.empty("policy_rules"),
        provenance=provenance,
        enforcement_mode=PolicyEnforcementMode.AUDIT_ONLY,
    )
    return RuntimePolicyBundle(
        require_human_on_critical=True,
        domain_fragments={fragment_id: payload},
        declarative_policy_runtime=runtime,
    )


def test_unknown_prompt_aborts_bind(tmp_path: Path) -> None:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(
            skill_id="skill.prompt",
            description="prompt",
            prompt_instruction_ids=("missing.prompt",),
        ),
    )
    composition = SkillResolver(registry).resolve_composition(["skill.prompt"])
    pinning = InMemorySkillExecutionPinningStore()
    host_bundle = RuntimePolicyBundle()
    with pytest.raises(SkillBridgeCompositionError, match="unknown prompt"):
        bind_resolved_skill_pack(
            tenant_id="t1",
            execution_id=mint_execution_id(),
            skill_profile=SkillProfile(enabled=["skill.prompt"]),
            skill_registry=registry,
            pinning_store=pinning,
            resolved_composition=composition,
            prompt_registry=YamlPromptRegistry(catalog_dir=tmp_path),
            host_policy_bundle=host_bundle,
        )


def test_conflicting_prompt_versions_abort_composition() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(
            skill_id="skill.a",
            description="a",
            prompt_instruction_ids=("demo.prompt@1",),
        ),
    )
    registry.register(
        SkillManifest(
            skill_id="skill.b",
            description="b",
            prompt_instruction_ids=("demo.prompt@2",),
        ),
    )
    with pytest.raises(SkillBridgeCompositionError, match="conflicting prompt versions"):
        SkillResolver(registry).resolve_composition(["skill.a", "skill.b"])


def test_resume_fails_when_registry_body_changes(tmp_path: Path) -> None:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(skill_id="skill.p", description="p", prompt_instruction_ids=("demo.prompt",)),
    )
    prompt_registry = _prompt_catalog(tmp_path, prompt_id="demo.prompt", version=1, body="alpha")
    composition = SkillResolver(registry).resolve_composition(["skill.p"])
    pinning = InMemorySkillExecutionPinningStore()
    binding = bind_resolved_skill_pack(
        tenant_id="t1",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["skill.p"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
        prompt_registry=prompt_registry,
        host_policy_bundle=RuntimePolicyBundle(),
    )
    prompt_path = tmp_path / "demo.prompt" / "1.yaml"
    text = prompt_path.read_text(encoding="utf-8").replace("alpha", "beta")
    prompt_path.write_text(text, encoding="utf-8")
    prompt_registry = YamlPromptRegistry(catalog_dir=tmp_path)
    prompt_registry.load_all()
    with pytest.raises(SkillBridgeCompositionError, match="digest mismatch"):
        resolve_pinned_prompt_instructions(binding.pinned_prompt_instructions, prompt_registry)


def test_fragment_outside_catalog_fails() -> None:
    pack = replace(
        SkillResolver(SkillRegistry()).resolve_composition([]).pack,
        policy_fragment_ids=frozenset({"missing.fragment"}),
    )
    bundle = RuntimePolicyBundle()
    with pytest.raises(SkillBridgeCompositionError, match="missing from host catalog"):
        merge_skill_policy_fragments(bundle, pack)


def test_conflicting_fragment_rules_fail() -> None:
    fragment_a = {
        "declarative_rules": [
            {
                "rule_id": "shared.rule",
                "handler_id": "builtin.allow_deny",
                "resource_kind": "tool",
                "resource_id": "tool.x",
                "action": "deny",
            },
        ],
    }
    fragment_b = {
        "declarative_rules": [
            {
                "rule_id": "shared.rule",
                "handler_id": "builtin.allow_deny",
                "resource_kind": "tool",
                "resource_id": "tool.x",
                "action": "allow",
            },
        ],
    }
    runtime = _policy_bundle_with_fragment("frag.a", fragment_a).declarative_policy_runtime
    assert runtime is not None
    bundle = RuntimePolicyBundle(
        require_human_on_critical=True,
        domain_fragments={"frag.a": fragment_a, "frag.b": fragment_b},
        declarative_policy_runtime=runtime,
    )
    pack = replace(
        SkillResolver(SkillRegistry()).resolve_composition([]).pack,
        policy_fragment_ids=frozenset({"frag.a", "frag.b"}),
    )
    with pytest.raises(SkillBridgeCompositionError, match="conflicting policy rule"):
        merge_skill_policy_fragments(bundle, pack)


def test_skill_cannot_weaken_host_policy() -> None:
    host_rule = DeclarativePolicyRule(
        rule_id="host.deny",
        handler_id="builtin.allow_deny",
        resource_kind="tool",
        resource_id="tool.x",
        action=PolicyRuleAction.DENY,
    )
    fragment_payload = {
        "declarative_rules": [
            {
                "rule_id": "host.deny",
                "handler_id": "builtin.allow_deny",
                "resource_kind": "tool",
                "resource_id": "tool.x",
                "action": "allow",
            },
        ],
    }
    bundle = _policy_bundle_with_fragment("frag.weaken", fragment_payload, host_rules=(host_rule,))
    pack = replace(
        SkillResolver(SkillRegistry()).resolve_composition([]).pack,
        policy_fragment_ids=frozenset({"frag.weaken"}),
    )
    with pytest.raises(SkillBridgeCompositionError, match="weakens host policy"):
        merge_skill_policy_fragments(bundle, pack)


def test_clearing_require_human_fails() -> None:
    bundle = _policy_bundle_with_fragment(
        "frag.hitl",
        {"require_human_on_critical": False},
    )
    pack = replace(
        SkillResolver(SkillRegistry()).resolve_composition([]).pack,
        policy_fragment_ids=frozenset({"frag.hitl"}),
    )
    with pytest.raises(SkillBridgeCompositionError, match="require_human_on_critical"):
        merge_skill_policy_fragments(bundle, pack)