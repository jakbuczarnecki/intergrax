# © Artur Czarnecki. All rights reserved.
"""T-IGX-051-A Part G: tenant-scoped reindex queue hooks."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from intergrax.rag.governance.embedding_version_policy import (
    ReindexQueueRequest,
    _notify_reindex_needed,
    clear_reindex_queue_hooks,
    evaluate_ingest_embedding_version,
    register_reindex_queue_hook,
)
from intergrax.rag.profiles.rag_profile import RagProfile

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-alpha"
_TENANT_B = "tenant-beta"


@pytest.fixture(autouse=True)
def _clear_hooks() -> Iterator[None]:
    clear_reindex_queue_hooks()
    yield
    clear_reindex_queue_hooks()


def _notify(*, tenant_id: str | None) -> bool:
    return _notify_reindex_needed(
        source_path="/data/doc.pdf",
        current_version="v1",
        target_version="v2",
        reason="test_scope",
        tenant_id=tenant_id,
    )


def test_ten_09a_hook_for_a_not_run_for_b() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(received.append, tenant_id=_TENANT_A)

    assert _notify(tenant_id=_TENANT_B) is False
    assert received == []


def test_ten_09b_hook_for_a_runs_for_a() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(received.append, tenant_id=_TENANT_A)

    assert _notify(tenant_id=_TENANT_A) is True
    assert len(received) == 1
    assert received[0].tenant_id == _TENANT_A


def test_ten_09c_tenantless_request_skips_scoped_hooks() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(received.append, tenant_id=_TENANT_A)

    assert _notify(tenant_id=None) is False
    assert received == []


def test_ten_09d_platform_wide_hook_unchanged() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(received.append)

    assert _notify(tenant_id=_TENANT_A) is True
    assert _notify(tenant_id=_TENANT_B) is True
    assert _notify(tenant_id=None) is True
    assert len(received) == 3


def test_ten_09e_failing_hook_isolated() -> None:
    healthy: list[ReindexQueueRequest] = []

    def _raising(_: ReindexQueueRequest) -> None:
        raise RuntimeError("scoped hook exploded")

    register_reindex_queue_hook(_raising, tenant_id=_TENANT_A)
    register_reindex_queue_hook(healthy.append, tenant_id=_TENANT_A)

    assert _notify(tenant_id=_TENANT_A) is True
    assert len(healthy) == 1


def test_ingest_forwards_tenant_into_reindex_request() -> None:
    received: list[ReindexQueueRequest] = []
    register_reindex_queue_hook(received.append, tenant_id=_TENANT_A)
    profile = RagProfile(embedding_model_version="v2", embedding_version_warn_on_ingest=True)

    evaluate_ingest_embedding_version(
        profile=profile,
        base_metadata={"embedding_model_version": "v1"},
        source_path="/data/corpus.pdf",
        tenant_id=_TENANT_A,
    )

    assert len(received) == 1
    assert received[0].tenant_id == _TENANT_A
