# © Artur Czarnecki. All rights reserved.

"""Backend-neutral scale probe contract for DIAG-FUNCTIONAL-SCALE-S1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from intergrax.integrations.contracts.document_store import ConditionalDocumentStore
from intergrax.knowledge.contracts.validation import JsonObject


@dataclass(frozen=True, slots=True)
class ScaleBackendIdentity:
    provider_id: str
    document_store_type: str
    database_name: str
    collection_name: str


@dataclass(frozen=True, slots=True)
class BackendIndexObservation:
    index_name: str
    keys: tuple[tuple[str, int], ...]
    unique: bool

    def to_json_dict(self) -> JsonObject:
        return {
            "index_name": self.index_name,
            "keys": [[key, direction] for key, direction in self.keys],
            "unique": self.unique,
        }


@dataclass(frozen=True, slots=True)
class BackendQueryEfficiencyObservation:
    partition_key: str
    row_key_prefix: str
    documents_examined: int
    keys_examined: int
    n_returned: int

    def to_json_dict(self) -> JsonObject:
        return {
            "partition_key": self.partition_key,
            "row_key_prefix": self.row_key_prefix,
            "documents_examined": self.documents_examined,
            "keys_examined": self.keys_examined,
            "n_returned": self.n_returned,
        }


@dataclass(frozen=True, slots=True)
class BackendResourceObservation:
    document_count: int | None
    storage_size_bytes: int | None
    indexes: tuple[BackendIndexObservation, ...]

    def to_json_dict(self) -> JsonObject:
        return {
            "document_count": self.document_count,
            "storage_size_bytes": self.storage_size_bytes,
            "indexes": [item.to_json_dict() for item in self.indexes],
        }


class ScaleBackendProbe(Protocol):
    @property
    def provider_id(self) -> str: ...

    def prepare(self) -> None: ...

    def build_document_store(self) -> ConditionalDocumentStore: ...

    def backend_identity(self) -> ScaleBackendIdentity: ...

    def collect_backend_metrics(self) -> BackendResourceObservation: ...

    def observe_execution_query_efficiency(
        self,
        *,
        tenant_id: str,
        task_id: str,
        run_id: str,
    ) -> BackendQueryEfficiencyObservation | None: ...

    def cleanup(self) -> None: ...

    def close_document_store(self, store: ConditionalDocumentStore) -> None: ...


@dataclass(frozen=True, slots=True)
class ScaleGateResult:
    gate_id: str
    passed: bool
    detail: str = ""

    def to_json_dict(self) -> JsonObject:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "detail": self.detail,
        }


class ScaleGate(Protocol):
    gate_id: str

    def evaluate(self) -> ScaleGateResult: ...


__all__ = [
    "BackendIndexObservation",
    "BackendQueryEfficiencyObservation",
    "BackendResourceObservation",
    "ScaleBackendIdentity",
    "ScaleBackendProbe",
    "ScaleGate",
    "ScaleGateResult",
]
