# © Artur Czarnecki. All rights reserved.

"""Typed configuration for cross-source policy strategies."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.context.policy.semantic_dedup import SemanticDedupPolicyConfig


@dataclass(frozen=True, slots=True)
class ContextPolicyPipelineConfig:
    semantic_dedup: SemanticDedupPolicyConfig = SemanticDedupPolicyConfig()
    strict_strategies: bool = False
