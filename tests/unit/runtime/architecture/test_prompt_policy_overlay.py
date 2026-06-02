from __future__ import annotations

from intergrax.runtime.architecture.prompt_composition import (
    PromptCompositionSpec,
    PromptLayer,
    PromptLayerFragment,
)
from intergrax.runtime.architecture.prompt_policy_overlay import (
    PolicyOverlayFragment,
    apply_policy_overlays,
)


def test_policy_overlay_appends_overlays_in_priority_order() -> None:
    report = apply_policy_overlays(
        spec=PromptCompositionSpec(
            prompt_id="prompt.test",
            version="1.0.0",
            fragments=[
                PromptLayerFragment(layer=PromptLayer.SYSTEM, content="System"),
                PromptLayerFragment(layer=PromptLayer.POLICY, content="Base policy"),
                PromptLayerFragment(layer=PromptLayer.TASK, content="Task"),
            ],
        ),
        overlays=[
            PolicyOverlayFragment(overlay_id="b", content="Overlay B", priority=20),
            PolicyOverlayFragment(overlay_id="a", content="Overlay A", priority=10),
        ],
    )
    assert len(report.overlay_trace) == 2
    assert "Overlay A" in report.composed_prompt.rendered_prompt
    assert "Overlay B" in report.composed_prompt.rendered_prompt
