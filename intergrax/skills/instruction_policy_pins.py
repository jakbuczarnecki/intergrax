# © Artur Czarnecki. All rights reserved.

"""Execution-scoped skill instruction and policy fragment pins (SK-BRIDGE)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from intergrax.contracts.validation import compute_sha256_content_digest
from intergrax.prompts.registry.pin_config import PromptPinConfig
from intergrax.prompts.registry.yaml_registry import YamlPromptRegistry
from intergrax.prompts.storage.models import LoadedPrompt, PromptNotFound
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.resolver import ResolvedSkillComposition, ResolvedSkillPack, SkillResolutionError

_PROMPT_VERSION_SUFFIX = re.compile(r"^(?P<prompt_id>.+)@(?P<version>\d+)$")


class SkillBridgeCompositionError(SkillResolutionError):
    """Skill prompt or policy composition is inconsistent or inadmissible."""


@dataclass(frozen=True, slots=True)
class SkillPromptInstructionPin:
    prompt_id: str
    version: int
    content_digest: str


@dataclass(frozen=True, slots=True)
class SkillPolicyFragmentPin:
    fragment_id: str
    content_digest: str


def _canonical_json_digest(payload: object) -> str:
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return compute_sha256_content_digest(encoded)


def parse_prompt_instruction_ref(ref: str) -> tuple[str, int | None]:
    normalized = ref.strip()
    if not normalized:
        raise SkillBridgeCompositionError("prompt instruction ref must be non-empty")
    match = _PROMPT_VERSION_SUFFIX.match(normalized)
    if match is None:
        return normalized, None
    return match.group("prompt_id"), int(match.group("version"))


def collect_prompt_version_requests(
    composition: ResolvedSkillComposition,
) -> dict[str, int]:
    requested: dict[str, int] = {}
    for manifest in composition.observed_manifests:
        for raw_ref in manifest.prompt_instruction_ids:
            prompt_id, explicit_version = parse_prompt_instruction_ref(raw_ref)
            if explicit_version is None:
                continue
            previous = requested.get(prompt_id)
            if previous is not None and previous != explicit_version:
                raise SkillBridgeCompositionError(
                    f"conflicting prompt versions for {prompt_id}: {previous} vs {explicit_version}",
                )
            requested[prompt_id] = explicit_version
    return requested


def resolve_skill_prompt_instruction_pins(
    pack: ResolvedSkillPack,
    prompt_registry: YamlPromptRegistry,
    *,
    composition: ResolvedSkillComposition | None = None,
) -> tuple[SkillPromptInstructionPin, ...]:
    if not pack.prompt_instruction_ids:
        return ()
    explicit_versions = collect_prompt_version_requests(composition) if composition is not None else {}
    pins: list[SkillPromptInstructionPin] = []
    for prompt_id in sorted(pack.prompt_instruction_ids):
        pin_config: PromptPinConfig | None = None
        if prompt_id in explicit_versions:
            pin_config = PromptPinConfig(
                pins={prompt_id: explicit_versions[prompt_id]},
                content_digests={},
            )
        try:
            loaded = prompt_registry.resolve(prompt_id, pin_config)
        except PromptNotFound as exc:
            raise SkillBridgeCompositionError(
                f"unknown prompt instruction for skill bind: {prompt_id}",
            ) from exc
        pins.append(
            SkillPromptInstructionPin(
                prompt_id=prompt_id,
                version=loaded.document.version,
                content_digest=loaded.content_hash,
            ),
        )
    return tuple(pins)


def prompt_pin_config_from_binding(
    pins: tuple[SkillPromptInstructionPin, ...],
) -> PromptPinConfig:
    return PromptPinConfig(
        pins={item.prompt_id: item.version for item in pins},
        content_digests={item.prompt_id: item.content_digest for item in pins},
    )


def resolve_pinned_prompt_instructions(
    pins: tuple[SkillPromptInstructionPin, ...],
    prompt_registry: YamlPromptRegistry,
) -> dict[str, LoadedPrompt]:
    if not pins:
        return {}
    pin_config = prompt_pin_config_from_binding(pins)
    resolved: dict[str, LoadedPrompt] = {}
    for item in pins:
        try:
            loaded = prompt_registry.resolve(item.prompt_id, pin_config)
        except PromptNotFound as exc:
            raise SkillBridgeCompositionError(
                f"prompt content digest mismatch on resume: {item.prompt_id}",
            ) from exc
        if loaded.content_hash != item.content_digest:
            raise SkillBridgeCompositionError(
                f"prompt content digest mismatch on resume: {item.prompt_id}",
            )
        resolved[item.prompt_id] = loaded
    return resolved


def host_policy_fragment_payload(
    bundle: RuntimePolicyBundle,
    fragment_id: str,
) -> dict[str, Any]:
    raw = bundle.domain_fragments.get(fragment_id)
    if raw is None:
        raise SkillBridgeCompositionError(
            f"policy fragment missing from host catalog: {fragment_id}",
        )
    if isinstance(raw, dict):
        return dict(raw)
    return {"value": raw}


def policy_fragment_content_digest(fragment_id: str, payload: dict[str, Any]) -> str:
    return _canonical_json_digest({"fragment_id": fragment_id, "payload": payload})


def collect_skill_policy_fragment_pins(
    bundle: RuntimePolicyBundle,
    pack: ResolvedSkillPack,
) -> tuple[SkillPolicyFragmentPin, ...]:
    if not pack.policy_fragment_ids:
        return ()
    pins: list[SkillPolicyFragmentPin] = []
    for fragment_id in sorted(pack.policy_fragment_ids):
        payload = host_policy_fragment_payload(bundle, fragment_id)
        pins.append(
            SkillPolicyFragmentPin(
                fragment_id=fragment_id,
                content_digest=policy_fragment_content_digest(fragment_id, payload),
            ),
        )
    return tuple(pins)


def verify_pinned_policy_fragments_on_resume(
    bundle: RuntimePolicyBundle,
    pins: tuple[SkillPolicyFragmentPin, ...],
) -> None:
    for item in pins:
        payload = host_policy_fragment_payload(bundle, item.fragment_id)
        digest = policy_fragment_content_digest(item.fragment_id, payload)
        if digest != item.content_digest:
            raise SkillBridgeCompositionError(
                f"policy fragment digest mismatch on resume: {item.fragment_id}",
            )


def apply_skill_bridge_at_bind(
    *,
    composition: ResolvedSkillComposition,
    prompt_registry: YamlPromptRegistry | None,
    host_policy_bundle: RuntimePolicyBundle | None,
) -> tuple[tuple[SkillPromptInstructionPin, ...], tuple[SkillPolicyFragmentPin, ...]]:
    pack = composition.pack
    prompt_pins = ()
    if pack.prompt_instruction_ids:
        if prompt_registry is None:
            raise SkillBridgeCompositionError(
                "prompt_registry required to bind skill prompt instructions",
            )
        prompt_pins = resolve_skill_prompt_instruction_pins(
            pack,
            prompt_registry,
            composition=composition,
        )
    policy_pins = ()
    if pack.policy_fragment_ids:
        if host_policy_bundle is None:
            raise SkillBridgeCompositionError(
                "host policy bundle required to bind skill policy fragments",
            )
        policy_pins = collect_skill_policy_fragment_pins(host_policy_bundle, pack)
    return prompt_pins, policy_pins


def verify_skill_bridge_on_resume(
    binding_pins_prompt: tuple[SkillPromptInstructionPin, ...],
    binding_pins_policy: tuple[SkillPolicyFragmentPin, ...],
    *,
    prompt_registry: YamlPromptRegistry | None,
    host_policy_bundle: RuntimePolicyBundle | None,
) -> dict[str, LoadedPrompt]:
    loaded: dict[str, LoadedPrompt] = {}
    if binding_pins_prompt:
        if prompt_registry is None:
            raise SkillBridgeCompositionError(
                "prompt_registry required to resume pinned skill prompts",
            )
        loaded = resolve_pinned_prompt_instructions(binding_pins_prompt, prompt_registry)
    if binding_pins_policy:
        if host_policy_bundle is None:
            raise SkillBridgeCompositionError(
                "host policy bundle required to resume pinned skill policy fragments",
            )
        verify_pinned_policy_fragments_on_resume(host_policy_bundle, binding_pins_policy)
    return loaded
