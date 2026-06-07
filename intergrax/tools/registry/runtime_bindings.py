# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime-bound tool dependency protocols (avoid Tier-0 ↔ UAEP import cycles)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from intergrax.contracts.memory_write_policy import MemoryWritePolicy


@runtime_checkable
class RunTraceReaderBinding(Protocol):
    """Structural binding for persisted run trace reads (``RunTraceReader``)."""

    def read_run(self, run_id: str, tenant_id: str) -> Any: ...

    def list_runs(self, tenant_id: str, *, limit: int = 50) -> List[Any]: ...


@runtime_checkable
class OnlineEvaluationRegistryBinding(Protocol):
    """Append-only harness evaluation registry (V-EVAL / W-OPS.11)."""

    def append(self, observation: Any) -> None: ...

    def list_observations(self) -> List[Any]: ...


@runtime_checkable
class KeyValueCacheListerBinding(Protocol):
    """Optional cache backend extension for ``cache.list_keys``."""

    def list_keys(self, tenant_id: str, *, prefix: str = "", limit: int = 100) -> List[str]: ...


@runtime_checkable
class VectorstoreIndexLifecycleBinding(Protocol):
    """Structural binding for RAG index lifecycle catalog tools."""

    def list_document_ids(self, *, limit: int = 100, offset: int = 0) -> List[str]: ...

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]: ...

    def list_collections(self) -> List[str]: ...

    def count(self) -> int: ...

    def search_by_metadata(
        self,
        *,
        conditions: Dict[str, Any],
        limit: int = 50,
    ) -> List[Dict[str, Any]]: ...

    def purge_collection(self, *, dry_run: bool = True, tenant_id: str = "") -> Dict[str, Any]: ...


@runtime_checkable
class HumanDecisionStoreBinding(Protocol):
    """Structural binding for read-only HITL decision catalog tools."""

    def list_escalations(self, tenant_id: str, *, limit: int = 50) -> List[Any]: ...

    def get_decision(self, decision_id: str, tenant_id: str) -> Any | None: ...

    def summarize_queue(self, tenant_id: str) -> Dict[str, int]: ...


@runtime_checkable
class WebSearchCacheBinding(Protocol):
    """Structural binding for ``websearch.invalidate_cache``."""

    def invalidate_query_cache(self, *, query: str = "", clear_all: bool = False) -> int: ...


@runtime_checkable
class SessionStorageBinding(Protocol):
    """Structural binding for interaction session read tools."""

    def list_sessions(self, tenant_id: str, user_id: str, *, limit: int = 20) -> List[Dict[str, str]]: ...

    def get_last_user_input(self, tenant_id: str, session_id: str) -> Optional[str]: ...


@runtime_checkable
class VectorStoreDocumentListerBinding(Protocol):
    """Optional vector store extension for ``rag.list_documents`` / ``rag.get_document``."""

    def list_document_ids(self, *, limit: int = 100, offset: int = 0) -> List[str]: ...

    def get_document(self, document_id: str) -> Optional[Dict[str, Any]]: ...


@runtime_checkable
class TaskMemoryViewBinding(Protocol):
    """Structural binding for policy-scoped task memory (``PolicyScopedMemoryView``)."""

    async def read(self, namespace: str, key: str) -> Optional[Dict[str, Any]]: ...

    async def write(
        self,
        namespace: str,
        key: str,
        value: Dict[str, Any],
        *,
        policy: MemoryWritePolicy = MemoryWritePolicy.REPLACE,
    ) -> None: ...

    async def list(self, namespace: str, prefix: str = "") -> List[Any]: ...
