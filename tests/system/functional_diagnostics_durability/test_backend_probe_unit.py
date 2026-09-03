# © Artur Czarnecki. All rights reserved.

"""D1-R1-G — backend plugin abstraction unit proof for durability orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.integrations._shared.conformance import assert_conditional_document_store
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from tests.system.functional_diagnostics_durability.durable_backend import DurableBackendIdentity
from tests.system.functional_diagnostics_durability.durability_orchestrator import (
  DurabilityProcessProbe,
)

pytestmark = pytest.mark.unit


@dataclass(frozen=True, slots=True)
class _SyntheticDurableBackendProbe:
  collection_name: str

  @property
  def provider_id(self) -> str:
    return "synthetic-in-memory"

  def build_document_store(self) -> ConditionalDocumentStore:
    return assert_conditional_document_store(InMemoryDocumentStore())

  def backend_identity(self) -> DurableBackendIdentity:
    return DurableBackendIdentity(
      provider_id=self.provider_id,
      document_store_type="InMemoryDocumentStore",
      database_name="synthetic",
      collection_name=self.collection_name,
    )

  def close_document_store(self, store: ConditionalDocumentStore) -> None:
    store.close()


class _SyntheticDurabilityProcessProbe(DurabilityProcessProbe):
  def __init__(self, *, work_dir: Path, collection_name: str) -> None:
    super().__init__(work_dir=work_dir, collection_name=collection_name)
    self._synthetic_probe = _SyntheticDurableBackendProbe(collection_name=collection_name)

  def _env(self) -> dict[str, str]:
    env = super()._env()
    env["DIAG_D1_R1_SYNTHETIC_BACKEND"] = "1"
    return env


def test_durable_backend_probe_protocol_is_provider_neutral() -> None:
  probe = _SyntheticDurableBackendProbe(collection_name="synthetic_collection")
  identity = probe.backend_identity()
  store = probe.build_document_store()
  try:
    assert identity.provider_id == "synthetic-in-memory"
    assert identity.collection_name == "synthetic_collection"
    assert isinstance(store, ConditionalDocumentStore)
  finally:
    probe.close_document_store(store)


def test_durability_process_probe_env_contains_mongodb_uri_only(tmp_path: Path) -> None:
  orchestrator = DurabilityProcessProbe(
    work_dir=tmp_path,
    collection_name="synthetic_collection",
  )
  env = orchestrator._env()
  assert "INTERGRAX_MONGODB_URI" in env
  assert "INTERGRAX_MONGODB_COLLECTION" in env
  assert env["INTERGRAX_MONGODB_COLLECTION"] == "synthetic_collection"
