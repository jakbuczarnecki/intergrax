from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import pytest

from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreRecord,
    VectorStoreScope,
)
from intergrax.rag.vectorstore.contracts.vector_store import VectorStore
from tests.integration.rag.vectorstore.test_vectorstore_real_backends import (
    _Backend,
    _is_known_environment_failure,
    _run_full_lifecycle,
    _skip_or_raise_backend_failure,
)

pytestmark = pytest.mark.unit


def test_known_unavailable_backend_condition_is_skippable() -> None:
    error = ConnectionRefusedError("connection refused")

    assert _is_known_environment_failure("qdrant", error) is True
    with pytest.raises(pytest.skip.Exception):
        _skip_or_raise_backend_failure("qdrant", error)

    assert _is_known_environment_failure("qdrant", RuntimeError("native ABI mismatch")) is False


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
