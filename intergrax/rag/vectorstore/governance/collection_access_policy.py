# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Optional collection-level access policy for vector stores (M-RAG.65)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

CollectionOperation = Literal["read", "write"]


class CollectionAccessDenied(PermissionError):
    """Raised when a collection operation violates ``CollectionAccessPolicy``."""


@dataclass(frozen=True)
class CollectionAccessPolicy:
    """
    Fine-grained access control for vector collection operations.

    Pair with UAEP / ``ApplicationSecurityProfile`` at Tier-3 wiring time.
    """

    tenant_id: str
    allowed_workspace_ids: Optional[frozenset[str]] = None
    deny_query: bool = False
    deny_ingest: bool = False
    read_only_collections: frozenset[str] = frozenset()


def enforce_collection_access(
    policy: Optional[CollectionAccessPolicy],
    operation: CollectionOperation,
    *,
    workspace_id: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> None:
    """Raise ``CollectionAccessDenied`` when the operation is not permitted."""
    if policy is None:
        return

    if operation == "read" and policy.deny_query:
        raise CollectionAccessDenied("collection_query_denied_by_policy")

    if operation == "write" and policy.deny_ingest:
        raise CollectionAccessDenied("collection_ingest_denied_by_policy")

    if (
        collection_name
        and collection_name in policy.read_only_collections
        and operation == "write"
    ):
        raise CollectionAccessDenied(f"collection_read_only:{collection_name}")

    if policy.allowed_workspace_ids is not None:
        if workspace_id is None or workspace_id not in policy.allowed_workspace_ids:
            raise CollectionAccessDenied("workspace_not_allowed_for_collection")
