# © Artur Czarnecki. All rights reserved.

"""Typed routing profile identities for DIAG-FUNCTIONAL-Q4."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.llm_adapters.routing.evaluator import profile_identity

Q4_PROFILE_A_MODEL = "llama3.1:latest"
Q4_PROFILE_B_MODEL = "qwen2.5:7b"
Q4_INVOKE_FAIL_MODEL = "diag-q4-nonexistent-model-xyz"

Q4_PRIMARY_TASK_CLASS = "model_routing_primary"
Q4_INVOKE_FAIL_TASK_CLASS = "model_routing_invoke_fail"


@dataclass(frozen=True, slots=True)
class RoutingProfileCandidate:
    rank: int
    profile: LLMProfile


def build_profile_a() -> LLMProfile:
    return LLMProfile(provider=LLMProvider.OLLAMA, model=Q4_PROFILE_A_MODEL)


def build_profile_b() -> LLMProfile:
    return LLMProfile(provider=LLMProvider.OLLAMA, model=Q4_PROFILE_B_MODEL)


def build_invoke_fail_profile() -> LLMProfile:
    return LLMProfile(provider=LLMProvider.OLLAMA, model=Q4_INVOKE_FAIL_MODEL)


def artifact_ref_for_profile(profile: LLMProfile) -> str:
    return f"llm:{profile_identity(profile)}"


def candidates_from_profiles(profiles: tuple[LLMProfile, ...]) -> tuple[RoutingProfileCandidate, ...]:
    return tuple(
        RoutingProfileCandidate(rank=index + 1, profile=profile)
        for index, profile in enumerate(profiles)
    )


__all__ = [
    "Q4_INVOKE_FAIL_MODEL",
    "Q4_INVOKE_FAIL_TASK_CLASS",
    "Q4_PRIMARY_TASK_CLASS",
    "Q4_PROFILE_A_MODEL",
    "Q4_PROFILE_B_MODEL",
    "RoutingProfileCandidate",
    "artifact_ref_for_profile",
    "build_invoke_fail_profile",
    "build_profile_a",
    "build_profile_b",
    "candidates_from_profiles",
]
