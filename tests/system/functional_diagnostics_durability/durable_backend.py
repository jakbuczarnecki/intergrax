# © Artur Czarnecki. All rights reserved.

"""Backend-neutral durable DocumentStore probe contract for D1-R1 qualification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore


@dataclass(frozen=True, slots=True)
class DurableBackendIdentity:
  provider_id: str
  document_store_type: str
  database_name: str
  collection_name: str


class DurableBackendProbe(Protocol):
  @property
  def provider_id(self) -> str: ...

  def build_document_store(self) -> ConditionalDocumentStore: ...

  def backend_identity(self) -> DurableBackendIdentity: ...

  def close_document_store(self, store: ConditionalDocumentStore) -> None: ...


__all__ = [
  "DurableBackendIdentity",
  "DurableBackendProbe",
]
