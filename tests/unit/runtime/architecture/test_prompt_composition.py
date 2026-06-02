from __future__ import annotations

import pytest

from intergrax.runtime.architecture.prompt_composition import (
    PromptCompositionSpec,
    PromptLayer,
    PromptLayerFragment,
    compose_prompt,
)


def test_compose_prompt_orders_layers_deterministically() -> None:
    composed = compose_prompt(
        PromptCompositionSpec(
            prompt_id="prompt.research",
            version="1.0.0",
            fragments=[
                PromptLayerFragment(layer=PromptLayer.TASK, content="Do research."),
                PromptLayerFragment(layer=PromptLayer.SYSTEM, content="You are helpful."),
                PromptLayerFragment(layer=PromptLayer.POLICY, content="Follow policy."),
            ],
        )
    )
    assert composed.layer_order[0] == PromptLayer.SYSTEM
    assert "[system]" in composed.rendered_prompt
    assert "[policy]" in composed.rendered_prompt


def test_compose_prompt_requires_core_layers() -> None:
    with pytest.raises(ValueError, match="Missing required prompt layers"):
        PromptCompositionSpec(
            prompt_id="prompt.incomplete",
            version="1.0.0",
            fragments=[
                PromptLayerFragment(layer=PromptLayer.SYSTEM, content="System only."),
            ],
        )
