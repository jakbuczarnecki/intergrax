# © Artur Czarnecki. All rights reserved.

"""Deterministic policy injection overlays for prompt composition (Phase V-PE.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.prompt_composition import (
    ComposedPrompt,
    PromptCompositionSpec,
    PromptLayer,
    PromptLayerFragment,
    compose_prompt,
)


class PolicyOverlayFragment(BaseModel):
    overlay_id: str
    content: str
    priority: int = 100


class PolicyOverlayTraceRecord(BaseModel):
    overlay_id: str
    applied: bool
    reason: str = ""


class PolicyOverlayCompositionReport(BaseModel):
    schema_version: str = "1.0.0"
    composed_prompt: ComposedPrompt
    overlay_trace: list[PolicyOverlayTraceRecord] = Field(default_factory=list)


def apply_policy_overlays(
    *,
    spec: PromptCompositionSpec,
    overlays: list[PolicyOverlayFragment],
) -> PolicyOverlayCompositionReport:
    sorted_overlays = sorted(overlays, key=lambda item: item.priority)
    overlay_trace: list[PolicyOverlayTraceRecord] = []
    policy_fragments = [fragment for fragment in spec.fragments if fragment.layer == PromptLayer.POLICY]
    other_fragments = [fragment for fragment in spec.fragments if fragment.layer != PromptLayer.POLICY]
    merged_policy_content = "\n".join(fragment.content.strip() for fragment in policy_fragments)
    for overlay in sorted_overlays:
        merged_policy_content = f"{merged_policy_content}\n{overlay.content.strip()}"
        overlay_trace.append(
            PolicyOverlayTraceRecord(
                overlay_id=overlay.overlay_id,
                applied=True,
                reason="Overlay appended to policy layer in priority order",
            )
        )
    merged_spec = PromptCompositionSpec(
        prompt_id=spec.prompt_id,
        version=spec.version,
        fragments=[
            *other_fragments,
            PromptLayerFragment(
                layer=PromptLayer.POLICY,
                content=merged_policy_content,
                source_ref="policy/overlays",
            ),
        ],
    )
    composed = compose_prompt(merged_spec)
    return PolicyOverlayCompositionReport(composed_prompt=composed, overlay_trace=overlay_trace)
