# © Artur Czarnecki. All rights reserved.

"""Authoritative resolution of product Ask scope into retrieval scope."""

from __future__ import annotations

from local_workspace_application.workspaces.knowledge_ask_scope_models import (
    KnowledgeAskScopeError,
    KnowledgeAskScopeV1,
    KnowledgeRetrievalScopeV1,
)
from local_workspace_application.workspaces.knowledge_inspection_operations_service import (
    KnowledgeAccessModeV1,
    KnowledgeInspectionService,
    KnowledgeInventoryError,
    KnowledgeInventoryItemV1,
)


class KnowledgeAskScopeResolver:
    """Resolve canonical knowledge_item_ids into validated source membership scope."""

    def __init__(self, inspection_service: KnowledgeInspectionService) -> None:
        self._inspection = inspection_service

    def resolve(
        self,
        *,
        tenant_id: str,
        workspace_id: str,
        scope: KnowledgeAskScopeV1,
    ) -> KnowledgeRetrievalScopeV1:
        if not scope.knowledge_item_ids:
            raise KnowledgeAskScopeError("knowledge_ask_scope_empty")

        resolved_items: list[KnowledgeInventoryItemV1] = []
        for knowledge_item_id in scope.knowledge_item_ids:
            try:
                item = self._inspection.get_item(
                    tenant_id=tenant_id,
                    workspace_id=workspace_id,
                    knowledge_item_id=knowledge_item_id,
                )
            except KnowledgeInventoryError as exc:
                raise self._map_inventory_error(exc.error_code) from exc
            resolved_items.append(item)

        indexed_items = [
            item for item in resolved_items if item.mode is KnowledgeAccessModeV1.INDEXED
        ]
        live_items = [
            item for item in resolved_items if item.mode is KnowledgeAccessModeV1.LIVE
        ]
        if live_items:
            if indexed_items:
                raise KnowledgeAskScopeError("knowledge_ask_scope_live_unsupported")
            raise KnowledgeAskScopeError("knowledge_ask_scope_live_unsupported")

        source_ids: list[str] = []
        for item in indexed_items:
            if item.tenant_id != tenant_id or item.workspace_id != workspace_id:
                raise KnowledgeAskScopeError("knowledge_ask_scope_not_found")
            if item.detached:
                raise KnowledgeAskScopeError("knowledge_ask_scope_detached")
            if not item.enabled:
                raise KnowledgeAskScopeError("knowledge_ask_scope_disabled")
            source_id = str(item.source_id or "").strip()
            if not source_id:
                raise KnowledgeAskScopeError("knowledge_ask_scope_invalid")
            source_ids.append(source_id)

        return KnowledgeRetrievalScopeV1.from_validated_source_ids(tuple(source_ids))

    @staticmethod
    def _map_inventory_error(error_code: str) -> KnowledgeAskScopeError:
        if error_code == "knowledge_item_not_found":
            return KnowledgeAskScopeError("knowledge_ask_scope_not_found")
        return KnowledgeAskScopeError("knowledge_ask_scope_invalid")
