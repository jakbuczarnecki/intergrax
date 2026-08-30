# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.contracts.instrumentation_span_attributes import (
    INTERGRAX_ATTEMPT_ID_ATTR,
    INTERGRAX_EXECUTION_ID_ATTR,
    INTERGRAX_RUN_ID_ATTR,
    active_execution_span_attributes,
    is_safe_instrumentation_span_attribute_key,
    merge_safe_span_attributes,
    normalize_span_attribute_value,
)

pytestmark = pytest.mark.unit


def test_active_execution_span_attributes_reads_bound_identity() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    try:
        attributes = active_execution_span_attributes()
    finally:
        reset_active_execution_identity(token)

    assert attributes == {
        INTERGRAX_RUN_ID_ATTR: str(run_id),
        INTERGRAX_ATTEMPT_ID_ATTR: str(attempt_id),
        INTERGRAX_EXECUTION_ID_ATTR: str(execution_id),
    }


def test_active_execution_span_attributes_never_mints_synthetic_ids() -> None:
    assert active_execution_span_attributes() == {}


def test_merge_safe_span_attributes_accepts_scalar_attributes() -> None:
    merged = merge_safe_span_attributes(
        caller_attributes={
            "rag.query.length": 12,
            "rag.ingest.dual_index": True,
            "rag.ingest.num_chunks": 3,
            "rag.tenant_id": "tenant-a",
        },
        include_active_identity=False,
    )

    assert merged == {
        "rag.query.length": 12,
        "rag.ingest.dual_index": True,
        "rag.ingest.num_chunks": 3,
        "rag.tenant_id": "tenant-a",
    }


def test_normalize_span_attribute_value_preserves_scalars() -> None:
    assert normalize_span_attribute_value("tenant-a") == "tenant-a"
    assert normalize_span_attribute_value(12) == 12
    assert normalize_span_attribute_value(True) is True
    assert normalize_span_attribute_value(None) is None


def test_merge_safe_span_attributes_drops_raw_content_keys() -> None:
    merged = merge_safe_span_attributes(
        caller_attributes={
            "rag.query.length": 12,
            "rag.query": "secret prompt",
            "rag.chunk.content": "body",
            "rag.tenant_id": "tenant-a",
        },
        include_active_identity=False,
    )

    assert merged == {
        "rag.query.length": 12,
        "rag.tenant_id": "tenant-a",
    }


@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("rag.query.length", True),
        ("rag.query", False),
        ("prompt.text", False),
        ("rag.ingest.num_chunks", True),
        ("intergrax.run_id", True),
    ],
)
def test_is_safe_instrumentation_span_attribute_key(key: str, expected: bool) -> None:
    assert is_safe_instrumentation_span_attribute_key(key) is expected
