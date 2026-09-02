"""Typed Qdrant client surface for VPI bootstrap."""

from __future__ import annotations

from typing import Protocol


class QdrantBootstrapClient(Protocol):
    def get_collections(self) -> object: ...

    def get_collection(self, collection_name: str) -> object: ...

    def create_collection(self, *args, **kwargs) -> bool: ...

    def upsert(self, *args, **kwargs) -> object: ...

    def close(self) -> None: ...
