# © Artur Czarnecki. All rights reserved.

"""Typed sparse profile deltas for layered resolution (P1.1)."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from intergrax.applications.contracts.profile_resolution.layer import ProfileLayer

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.llm_adapters.registry.profile import LLMProfile
from intergrax.tools.registry.profile import ToolProfile

from intergrax.applications.contracts.environment_profile.sub_profiles import CostProfile

T = TypeVar("T")


class ProfileFieldUpdate(BaseModel, Generic[T]):
    """Sparse field opinion — absent parent field means no opinion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["set", "clear"] = "set"
    value: T | None = None

    @model_validator(mode="after")
    def _validate_set_requires_value(self) -> ProfileFieldUpdate[T]:
        if self.action == "set" and self.value is None:
            raise ValueError("set action requires a non-null value")
        return self


class ProfileDelta(BaseModel):
    """Sparse requested changes for one overlay layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    llm_profile: ProfileFieldUpdate[LLMProfile] | None = None
    tool_profile: ProfileFieldUpdate[ToolProfile] | None = None
    execution_mode: ProfileFieldUpdate[ExecutionMode] | None = None
    cost_profile: ProfileFieldUpdate[CostProfile] | None = None

    def opinion_paths(self) -> tuple[str, ...]:
        """Return canonical dotted paths with an explicit opinion in this delta."""
        paths: list[str] = []
        if self.llm_profile is not None:
            paths.append("capabilities.llm")
        if self.tool_profile is not None:
            paths.append("capabilities.tools")
        if self.execution_mode is not None:
            paths.append("meta.execution_mode")
        if self.cost_profile is not None:
            paths.append("governance.cost")
        return tuple(paths)


class ProfileLayerInput(BaseModel):
    """One overlay contribution submitted to profile resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: ProfileLayer
    revision: str | None = None
    delta: ProfileDelta | None = None

    @model_validator(mode="after")
    def _validate_layer_and_payload(self) -> ProfileLayerInput:
        if self.layer == ProfileLayer.APPLICATION:
            raise ValueError("application layer is supplied via application_profile, not layer inputs")
        if self.delta is None:
            raise ValueError("layer input requires a typed delta")
        if not self.delta.opinion_paths():
            raise ValueError("delta must express at least one field opinion")
        return self
