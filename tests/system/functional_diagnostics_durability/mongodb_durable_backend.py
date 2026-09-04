# © Artur Czarnecki. All rights reserved.

"""MongoDB durable backend probe for D1-R1 qualification."""

from __future__ import annotations

import os

from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.integrations.providers.document_store.mongodb.bundle import (
  create_mongodb_document_store,
)
from intergrax.integrations.providers.document_store.mongodb.config import (
  ENV_MONGODB_COLLECTION,
  ENV_MONGODB_DATABASE,
  ENV_MONGODB_URI,
  MongoDBIntegrationConfig,
)
from tests.system.functional_diagnostics_durability.durable_backend import (
  DurableBackendIdentity,
  DurableBackendProbe,
)

_DEFAULT_URI = "mongodb://localhost:27017"
_DEFAULT_DATABASE = "intergrax_diag_d1_r1"


class MongoDurableBackendProbe:
  """Production MongoDB DocumentStore provider probe."""

  def __init__(self, *, collection_name: str) -> None:
    self._collection_name = collection_name
    self._config = MongoDBIntegrationConfig.from_env(
      database=_DEFAULT_DATABASE,
      collection_name=collection_name,
    )

  @property
  def provider_id(self) -> str:
    return "mongodb"

  def build_document_store(self) -> ConditionalDocumentStore:
    uri = os.environ.get(ENV_MONGODB_URI, _DEFAULT_URI).strip() or _DEFAULT_URI
    if not uri:
      raise IntegrationConfigurationError("INTERGRAX_MONGODB_URI is required for D1-R1")
    store = create_mongodb_document_store(
      uri=uri,
      database=self._config.database,
      collection_name=self._config.collection_name,
    )
    return assert_conditional_document_store(store)

  def backend_identity(self) -> DurableBackendIdentity:
    database, collection = self._config.qualified_collection()
    return DurableBackendIdentity(
      provider_id=self.provider_id,
      document_store_type="_MongoDBDocumentStore",
      database_name=database,
      collection_name=collection,
    )

  def close_document_store(self, store: ConditionalDocumentStore) -> None:
    store.close()


def resolve_mongodb_uri() -> str:
  return os.environ.get(ENV_MONGODB_URI, _DEFAULT_URI).strip() or _DEFAULT_URI


def mongodb_available() -> bool:
  uri = resolve_mongodb_uri()
  return bool(uri.strip())


__all__ = [
  "MongoDurableBackendProbe",
  "mongodb_available",
  "resolve_mongodb_uri",
]
