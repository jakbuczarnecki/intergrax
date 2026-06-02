# © Artur Czarnecki. All rights reserved.

"""Prompt composition model for layered prompt assembly (Phase V-PE.2)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class PromptLayer(str, Enum):
    SYSTEM = "system"
    TASK = "task"
    POLICY = "policy"
    CONTEXT = "context"


class PromptLayerFragment(BaseModel):
    layer: PromptLayer
    content: str
    source_ref: str = ""


class PromptCompositionSpec(BaseModel):
    prompt_id: str
    version: str
    fragments: list[PromptLayerFragment] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_required_layers(self) -> "PromptCompositionSpec":
        present_layers = {fragment.layer for fragment in self.fragments}
        required = {PromptLayer.SYSTEM, PromptLayer.TASK, PromptLayer.POLICY}
        missing = sorted(layer.value for layer in required - present_layers)
        if missing:
            raise ValueError(f"Missing required prompt layers: {', '.join(missing)}")
        return self


class ComposedPrompt(BaseModel):
    prompt_id: str
    version: str
    rendered_prompt: str
    layer_order: list[PromptLayer] = Field(default_factory=list)


def compose_prompt(spec: PromptCompositionSpec) -> ComposedPrompt:
    ordered_layers = [
        PromptLayer.SYSTEM,
        PromptLayer.POLICY,
        PromptLayer.TASK,
        PromptLayer.CONTEXT,
    ]
    fragments_by_layer = {fragment.layer: fragment for fragment in spec.fragments}
    sections: list[str] = []
    layer_order: list[PromptLayer] = []
    for layer in ordered_layers:
        fragment = fragments_by_layer.get(layer)
        if fragment is None:
            continue
        layer_order.append(layer)
        sections.append(f"[{layer.value}]\n{fragment.content.strip()}")
    rendered = "\n\n".join(sections)
    return ComposedPrompt(
        prompt_id=spec.prompt_id,
        version=spec.version,
        rendered_prompt=rendered,
        layer_order=layer_order,
    )
