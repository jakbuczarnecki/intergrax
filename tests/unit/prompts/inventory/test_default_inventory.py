# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from intergrax.prompts.inventory.default_inventory import build_default_inventory
from intergrax.prompts.inventory.models import PromptInstructionKind


def test_inventory_contains_core_categories() -> None:
    inv = build_default_inventory()

    assert inv.by_kind(PromptInstructionKind.PLANNER)
    assert inv.by_kind(PromptInstructionKind.RAG_POLICY)
    assert inv.by_kind(PromptInstructionKind.HISTORY_SUMMARY)
    assert inv.by_kind(PromptInstructionKind.SYSTEM_BEHAVIOR)
