# Reference session turn index plugin for gate tests (Phase MEM-VEC-3.1).

from __future__ import annotations

from intergrax.memory.contracts.session_turn_index import SessionTurnIndexStoreCreationContext
from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore


class ExternalSessionTurnIndexStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.session_turn_index"

    @classmethod
    def create_session_turn_index(
        cls,
        context: SessionTurnIndexStoreCreationContext,
    ) -> VectorSessionTurnIndexStore:
        if context.embedding_manager is None or context.vectorstore_manager is None:
            raise ValueError("embedding_manager and vectorstore_manager are required")
        return VectorSessionTurnIndexStore(
            embedding_manager=context.embedding_manager,
            vectorstore_manager=context.vectorstore_manager,
            index_roles=context.index_roles,
            tenant_id=context.tenant_id,
            vector_index_namespace=context.vector_index_namespace,
            workspace_id=context.workspace_id,
        )
