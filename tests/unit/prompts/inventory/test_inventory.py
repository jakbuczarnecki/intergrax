# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.prompts.inventory.models import PromptInstructionKind
from intergrax.prompts.inventory.registry import PromptInventoryBuilder


def test_inventory_builder_groups_by_kind() -> None:
    builder = PromptInventoryBuilder()

    builder.add(
        kind=PromptInstructionKind.PLANNER,
        module="a.b",
        symbol="X",
        description="desc",
    )

    builder.add(
        kind=PromptInstructionKind.RAG_POLICY,
        module="c.d",
        symbol="Y",
        description="desc2",
    )

    inventory = builder.build()

    planners = inventory.by_kind(PromptInstructionKind.PLANNER)
    assert len(planners) == 1

    rags = inventory.by_kind(PromptInstructionKind.RAG_POLICY)
    assert len(rags) == 1
