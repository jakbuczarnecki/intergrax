# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Frozen legacy Critic contracts for migration historical evidence only (DS-MIG-04).

Not executable verification authority. Used by retirement qualification records and
historical trace readers after ``intergrax/runtime/critic/**`` deletion.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class LegacyCriticScope(str, Enum):
    NODE_PARTIAL = "node_partial"
    GRAPH_FINAL = "graph_final"
    UAEP_STEP = "uaep_step"
    OFFLINE_CASE = "offline_case"


class LegacyCriticLayer(str, Enum):
    L0_DETERMINISTIC = "l0_deterministic"
    L1_SEMANTIC = "l1_semantic"
    L1_TRAJECTORY = "l1_trajectory"


class LegacyCriticAction(str, Enum):
    CONTINUE = "continue"
    RETRY = "retry"
    REVISE = "revise"
    FAIL = "fail"


class LegacyLayerVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    layer: LegacyCriticLayer
    passed: bool
    score: float | None = None
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class LegacyCriticVerdict(BaseModel):
    """Immutable legacy critic verdict shape for historical parity and trace parsing."""

    model_config = ConfigDict(extra="forbid")

    scope: LegacyCriticScope
    passed: bool
    layers: list[LegacyLayerVerdict] = Field(default_factory=list)
    recommended_action: LegacyCriticAction = LegacyCriticAction.CONTINUE
    failure_reasons: list[str] = Field(default_factory=list)
