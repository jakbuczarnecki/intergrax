# © Artur Czarnecki. All rights reserved.

"""Workspace-scoped vector cleanup via the VectorstoreManager abstraction."""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from intergrax.rag.vectorstore.contracts.native_vectorstore import VectorStoreScope

logger = logging.getLogger(__name__)

_PAGE_SIZE = 200
_MAX_PAGES = 100


class WorkspaceVectorCleanupPort(Protocol):
    def delete_workspace_vectors(self, *, tenant_id: str, workspace_id: str) -> int: ...


def _unwrap_vector_backend(manager: Any) -> Any:
    """Resolve the concrete RAG store under manager / integration / bridge wrappers."""
    current = manager
    seen: set[int] = set()
    for _ in range(6):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        if callable(getattr(current, "search_by_metadata", None)) and callable(
            getattr(current, "delete", None)
        ):
            # Prefer an inner rag store when the outer object only forwards VectorStore basics.
            inner = getattr(current, "rag_store", None)
            if inner is not None and inner is not current:
                if callable(inner) and not isinstance(inner, type):
                    try:
                        resolved = inner()
                    except TypeError:
                        resolved = inner
                else:
                    resolved = inner
                if resolved is not None and resolved is not current:
                    current = resolved
                    continue
            nested = getattr(current, "_inner", None) or getattr(current, "_store", None)
            if nested is not None and nested is not current:
                current = nested
                continue
            return current
        nested = (
            getattr(current, "rag_store", None)
            or getattr(current, "_inner", None)
            or getattr(current, "_store", None)
        )
        if callable(nested) and not isinstance(nested, type):
            try:
                nested = nested()
            except TypeError:
                pass
        if nested is None or nested is current:
            break
        current = nested
    return current


def _bound_store_tenant(backend: Any) -> str | None:
    """Return the vector backend's fixed tenant when present (Qdrant/InMemory)."""
    if backend is None:
        return None
    cfg = getattr(backend, "cfg", None)
    for candidate in (
        getattr(cfg, "tenant_id", None) if cfg is not None else None,
        getattr(backend, "_tenant_id", None),
        getattr(backend, "tenant_id", None),
    ):
        text = str(candidate or "").strip()
        if text:
            return text
    return None


def _is_absent_collection_error(exc: BaseException) -> bool:
    """True when the vector backend has no collection yet (nothing to delete)."""
    name = type(exc).__name__
    msg = str(exc).casefold()
    if name in {"UnexpectedResponse", "NotFoundError", "NotFoundException"}:
        if "collection" in msg or "not found" in msg or "404" in msg:
            return True
    if "doesn't exist" in msg and "collection" in msg:
        return True
    if "not found: collection" in msg:
        return True
    return False


class VectorstoreManagerWorkspaceCleanup:
    """Delete vectors for one workspace within the process-bound vector store.

    Product ``tenant_id`` may differ from the vector backend's fixed tenant
    (e.g. Slack ``LOCAL_WORKSPACE_SLACK_TENANT_ID`` vs ``INTERGRAX_QDRANT_TENANT_ID``).
    Search uses the store-bound tenant when known, otherwise the cleanup
    request's tenant. Isolation within the store is by ``workspace_id`` /
    ``collection_id``.

    A missing Qdrant collection is treated as empty (0 vectors), not as failure —
    common for workspaces that never indexed.
    """

    def __init__(self, vectorstore_manager: Any) -> None:
        self._manager = vectorstore_manager

    def delete_workspace_vectors(self, *, tenant_id: str, workspace_id: str) -> int:
        manager = self._manager
        if manager is None:
            return 0
        workspace = (workspace_id or "").strip()
        if not workspace:
            return 0
        # ``tenant_id`` is part of the port for callers; vector backends that bind a
        # fixed tenant use that store tenant instead of the product tenant.

        search, delete, scope = self._resolve_ops(
            manager,
            tenant_id=tenant_id,
            workspace_id=workspace,
        )
        deleted = 0
        deleted_ids: set[str] = set()
        for field in ("workspace_id", "collection_id"):
            deleted += self._drain_matches(
                search=search,
                delete=delete,
                scope=scope,
                field=field,
                value=workspace,
                deleted_ids=deleted_ids,
            )
        return deleted

    def _resolve_ops(
        self,
        manager: Any,
        *,
        tenant_id: str,
        workspace_id: str,
    ) -> tuple[
        Callable[..., list[dict[str, Any]]],
        Callable[[list[str]], None],
        VectorStoreScope,
    ]:
        candidates: list[Any] = []
        unwrapped = _unwrap_vector_backend(manager)
        # Prefer concrete rag store over VectorstoreManager / integration wrappers:
        # manager.search_by_metadata often raises not_supported when the integration
        # does not implement VectorstoreIndexLifecycleBinding.
        for item in (unwrapped, getattr(manager, "_store", None), manager):
            if item is not None and item not in candidates:
                candidates.append(item)

        bound_scope = getattr(manager, "bound_scope", None)
        store_tenant = _bound_store_tenant(unwrapped)
        resolved_tenant = (
            getattr(bound_scope, "tenant_id", None)
            or store_tenant
            or tenant_id
        )
        scope = VectorStoreScope(
            tenant_id=str(resolved_tenant),
            namespace=getattr(bound_scope, "namespace", None),
            workspace_id=workspace_id,
        )
        native_delete = getattr(manager, "delete", None)
        if callable(native_delete):
            delete = self._make_scoped_delete(native_delete, scope)
        else:
            delete = None

        last_error: BaseException | None = None
        for target in candidates:
            search = getattr(target, "search_by_metadata", None)
            target_delete = getattr(target, "delete", None)
            if not callable(search) or (delete is None and not callable(target_delete)):
                continue
            probe: dict[str, Any] = {"workspace_id": "__probe__"}
            probe["tenant_id"] = scope.tenant_id
            try:
                search(conditions=probe, limit=1)
                if delete is None:
                    delete = self._make_scoped_delete(target_delete, scope)
                return search, delete, scope
            except RuntimeError as exc:
                # Wrapper without lifecycle binding — try next candidate.
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                if _is_absent_collection_error(exc):
                    if delete is None:
                        delete = self._make_scoped_delete(target_delete, scope)
                    return search, delete, scope
                last_error = exc
                continue

        logger.warning("workspace_vector_cleanup unsupported reason=missing_lifecycle_api")
        if last_error is not None:
            raise RuntimeError("vectorstore_workspace_cleanup_not_supported") from last_error
        raise RuntimeError("vectorstore_workspace_cleanup_not_supported")

    def _make_scoped_delete(
        self,
        delete: Callable[..., None],
        scope: VectorStoreScope,
    ) -> Callable[[list[str]], None]:
        def invoke(ids: list[str]) -> None:
            self._delete_scoped(delete, ids, scope=scope)

        return invoke

    @staticmethod
    def _delete_scoped(
        delete: Callable[..., None],
        ids: list[str],
        *,
        scope: VectorStoreScope,
    ) -> None:
        try:
            delete(ids, scope=scope)
        except TypeError as exc:
            raise RuntimeError("vectorstore_workspace_cleanup_not_supported") from exc

    def _drain_matches(
        self,
        *,
        search: Callable[..., list[dict[str, Any]]],
        delete: Callable[[list[str]], None],
        scope: VectorStoreScope,
        field: str,
        value: str,
        deleted_ids: set[str],
    ) -> int:
        deleted = 0
        conditions: dict[str, Any] = {field: value}
        conditions["tenant_id"] = scope.tenant_id
        for _ in range(_MAX_PAGES):
            try:
                matches = search(
                    conditions=conditions,
                    limit=_PAGE_SIZE,
                )
            except Exception as exc:  # noqa: BLE001
                if _is_absent_collection_error(exc):
                    logger.info(
                        "workspace_vector_cleanup absent_collection kind=%s",
                        type(exc).__name__,
                    )
                    return deleted
                logger.warning(
                    "workspace_vector_cleanup search_failed kind=%s",
                    type(exc).__name__,
                )
                raise RuntimeError("vectorstore_workspace_cleanup_not_supported") from exc
            if not matches:
                break
            ids = sorted(
                {
                    str(match.get("id") or "").strip()
                    for match in matches
                    if str(match.get("id") or "").strip()
                }
            )
            ids = [vector_id for vector_id in ids if vector_id not in deleted_ids]
            if not ids:
                break
            delete(ids)
            deleted_ids.update(ids)
            deleted += len(ids)
            if len(matches) < _PAGE_SIZE:
                break
        return deleted
