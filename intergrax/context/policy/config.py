# © Artur Czarnecki. All rights reserved.

"""Typed configuration for cross-source policy strategies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextPolicyPipelineConfig:
    strict_strategies: bool = False
