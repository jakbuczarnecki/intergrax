# © Artur Czarnecki. All rights reserved.

"""Shared knowledge inventory target resolution for conversation flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeInventoryItemV1,
)


@dataclass(frozen=True)
class KnowledgeTargetResolution:
    item: KnowledgeInventoryItemV1 | None
    ambiguous: bool = False


def normalize_knowledge_label(value: str) -> str:
    return " ".join(value.split()).casefold()


def resolve_knowledge_target(
    items: Sequence[KnowledgeInventoryItemV1],
    *,
    reference_kind: str,
    target_reference: str,
) -> KnowledgeTargetResolution:
    if reference_kind == "knowledge_item_id":
        matches = tuple(
            item for item in items if item.knowledge_item_id == target_reference.strip()
        )
        if len(matches) > 1:
            return KnowledgeTargetResolution(item=None, ambiguous=True)
        if len(matches) == 1:
            return KnowledgeTargetResolution(item=matches[0])
        return KnowledgeTargetResolution(item=None)
    if reference_kind == "ordinal":
        if not target_reference.isdigit():
            return KnowledgeTargetResolution(item=None)
        index = int(target_reference) - 1
        if index < 0 or index >= len(items):
            return KnowledgeTargetResolution(item=None)
        return KnowledgeTargetResolution(item=items[index])
    normalized = normalize_knowledge_label(target_reference)
    matches = tuple(
        item
        for item in items
        if item.display_label is not None
        and normalize_knowledge_label(item.display_label) == normalized
    )
    if len(matches) > 1:
        return KnowledgeTargetResolution(item=None, ambiguous=True)
    if len(matches) == 1:
        return KnowledgeTargetResolution(item=matches[0])
    return KnowledgeTargetResolution(item=None)
