# Reference session turn index plugin for gate tests (Phase MEM-VEC-3.1).

from __future__ import annotations

from typing import Any

from intergrax.memory.session_turn_index_service import VectorSessionTurnIndexStore


class ExternalSessionTurnIndexStorePlugin:
    @classmethod
    def plugin_id(cls) -> str:
        return "external.session_turn_index"

    @classmethod
    def create_session_turn_index(cls, **kwargs: Any) -> VectorSessionTurnIndexStore:
        return VectorSessionTurnIndexStore(
            embedding_manager=kwargs["embedding_manager"],
            vectorstore_manager=kwargs["vectorstore_manager"],
            index_roles=kwargs.get("index_roles", ("user", "assistant")),
            tenant_id=str(kwargs.get("tenant_id") or "default"),
            vector_index_namespace=kwargs.get("vector_index_namespace"),
        )
