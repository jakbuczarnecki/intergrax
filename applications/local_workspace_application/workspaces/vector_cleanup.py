# © Artur Czarnecki. All rights reserved.

"""Workspace-scoped vector cleanup via the VectorstoreManager abstraction."""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

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


class VectorstoreManagerWorkspaceCleanup:
    """Delete only vectors matching tenant + workspace (or collection_id=workspace)."""

    def __init__(self, vectorstore_manager: Any) -> None:
        self._manager = vectorstore_manager

    def delete_workspace_vectors(self, *, tenant_id: str, workspace_id: str) -> int:
        manager = self._manager
        if manager is None:
            return 0
        tenant = (tenant_id or "").strip()
        workspace = (workspace_id or "").strip()
        if not tenant or not workspace:
            return 0

        search, delete = self._resolve_ops(manager)
        deleted = 0
        for field in ("workspace_id", "collection_id"):
            deleted += self._drain_matches(
                search=search,
                delete=delete,
                tenant_id=tenant,
                field=field,
                value=workspace,
            )
        return deleted

    def _resolve_ops(
        self, manager: Any
    ) -> tuple[Callable[..., list[dict[str, Any]]], Callable[[list[str]], None]]:
        candidates: list[Any] = []
        unwrapped = _unwrap_vector_backend(manager)
        for item in (manager, getattr(manager, "_store", None), unwrapped):
            if item is not None and item not in candidates:
                candidates.append(item)

        last_error: BaseException | None = None
        for target in candidates:
            search = getattr(target, "search_by_metadata", None)
            delete = getattr(target, "delete", None)
            if not callable(search) or not callable(delete):
                continue
            try:
                search(
                    conditions={"tenant_id": "__probe__", "workspace_id": "__probe__"},
                    limit=1,
                )
                return search, delete
            except RuntimeError as exc:
                last_error = exc
                continue
            except ValueError:
                return search, delete
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        logger.warning("workspace_vector_cleanup unsupported reason=missing_lifecycle_api")
        if last_error is not None:
            raise RuntimeError("vectorstore_workspace_cleanup_not_supported") from last_error
        raise RuntimeError("vectorstore_workspace_cleanup_not_supported")

    def _drain_matches(
        self,
        *,
        search: Callable[..., list[dict[str, Any]]],
        delete: Callable[[list[str]], None],
        tenant_id: str,
        field: str,
        value: str,
    ) -> int:
        deleted = 0
        for _ in range(_MAX_PAGES):
            try:
                matches = search(
                    conditions={"tenant_id": tenant_id, field: value},
                    limit=_PAGE_SIZE,
                )
            except Exception as exc:  # noqa: BLE001
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
            if not ids:
                break
            delete(ids)
            deleted += len(ids)
            if len(matches) < _PAGE_SIZE:
                break
        return deleted
