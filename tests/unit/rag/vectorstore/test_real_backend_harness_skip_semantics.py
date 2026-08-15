from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from intergrax.integrations.contracts.base import IntegrationDependencyError
from intergrax.integrations.providers.vector_store.pgvector.rag_store import (
    PgVectorRagStore,
)
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from tests.integration.rag.vectorstore.test_vectorstore_real_backends import (
    _Backend,
    _record,
    _is_known_environment_failure,
    _run_full_lifecycle,
    _skip_or_raise_backend_failure,
)

pytestmark = pytest.mark.unit


class _FailingCursor:
    def __enter__(self) -> "_FailingCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, *_args: object) -> None:
        raise RuntimeError("SQL operation failed")


class _FailingConnection:
    def cursor(self) -> _FailingCursor:
        return _FailingCursor()

    def commit(self) -> None:
        return None

    def rollback(self) -> None:
        return None


def _failing_pgvector_store() -> PgVectorRagStore:
    store = object.__new__(PgVectorRagStore)
    store._tenant_id = "tenant_a"
    store._dimension = 4
    store._connection = _FailingConnection()
    return store


def test_known_unavailable_backend_condition_is_skippable() -> None:
    error = ConnectionRefusedError("connection refused")

    assert _is_known_environment_failure("qdrant", error, during_open=True) is True
    assert _is_known_environment_failure("pgvector", error, during_open=True) is True
    assert _is_known_environment_failure("pgvector", error, during_open=False) is False
    with pytest.raises(pytest.skip.Exception):
        _skip_or_raise_backend_failure("qdrant", error, during_open=True)

    assert (
        _is_known_environment_failure(
            "qdrant",
            RuntimeError("native ABI mismatch"),
            during_open=True,
        )
        is False
    )


def test_pgvector_missing_dependency_is_skippable_only_during_open() -> None:
    error = IntegrationDependencyError(
        "pgvector requires the integrations-pgvector extra"
    )

    assert _is_known_environment_failure("pgvector", error, during_open=True) is True
    assert _is_known_environment_failure("pgvector", error, during_open=False) is False
    with pytest.raises(IntegrationDependencyError):
        _skip_or_raise_backend_failure("pgvector", error, during_open=False)


@pytest.mark.parametrize(
    "operation",
    [
        "query",
        "add",
        "delete",
        "ownership",
        "count",
    ],
)
def test_pgvector_runtime_operation_failure_is_not_environment_skip(
    operation: str,
) -> None:
    store = _failing_pgvector_store()
    scope = VectorStoreScope(
        tenant_id="tenant_a",
        namespace="namespace_a",
        workspace_id="workspace_a",
    )

    with pytest.raises(RuntimeError, match="SQL operation failed") as raised:
        if operation == "query":
            store.query([1.0, 0.0, 0.0, 0.0], scope=scope, top_k=1)
        elif operation == "add":
            record = _record(
                "vector-1",
                source_id="source://runtime",
                scope=scope,
            )
            store.add_records([record], scope=scope)
        elif operation == "delete":
            store.delete(["vector-1"], scope=scope)
        elif operation == "ownership":
            store.list_source_record_ids(source_id="source://runtime", scope=scope)
        else:
            store.count(scope=scope)

    error = raised.value
    assert _is_known_environment_failure(
        "pgvector", error, during_open=False
    ) is False
    with pytest.raises(RuntimeError, match="SQL operation failed"):
        _skip_or_raise_backend_failure(
            "pgvector",
            error,
            during_open=False,
        )


def test_native_add_records_contract_failure_propagates() -> None:
    class _ContractFailingStore:
        def add_records(
            self,
            records: Sequence[VectorStoreRecord],
            *,
            scope: VectorStoreScope,
        ) -> Sequence[str]:
            del records, scope
            raise VectorStoreContractError("native ABI mismatch")

    backend = _Backend(
        slug="qdrant",
        store=cast(VectorStore, _ContractFailingStore()),
        scope=VectorStoreScope(
            tenant_id="tenant_a",
            namespace="namespace_a",
            workspace_id="workspace_a",
        ),
    )

    with pytest.raises(VectorStoreContractError, match="native ABI mismatch"):
        _run_full_lifecycle(backend)
