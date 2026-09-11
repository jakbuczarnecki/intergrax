# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionRecord,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageSegmentRecord,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.lineage.codecs import (
    decode_execution_lineage_admission_record,
    encode_execution_lineage_admission_record,
    encode_execution_lineage_attempt_scope,
)


def _scope() -> object:
    return build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )


def test_scope_rejects_blank_tenant() -> None:
    with pytest.raises(ValueError, match="tenant_id"):
        build_execution_lineage_attempt_scope(
            tenant_id="   ",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        )


def test_root_admission_record_validation() -> None:
    scope = _scope()
    root = mint_execution_id()
    record = ExecutionLineageAdmissionRecord(
        scope=scope,
        segment_root_execution_id=root,
        execution_id=root,
        parent_execution_id=None,
        admission_position=1,
    )
    assert record.parent_execution_id is None


def test_child_admission_record_rejects_parent_equal_execution() -> None:
    scope = _scope()
    execution = mint_execution_id()
    with pytest.raises(ValueError, match="execution_id != parent_execution_id"):
        ExecutionLineageAdmissionRecord(
            scope=scope,
            segment_root_execution_id=mint_execution_id(),
            execution_id=execution,
            parent_execution_id=execution,
            admission_position=2,
        )


def test_segment_record_roundtrip_codec() -> None:
    scope = _scope()
    root = mint_execution_id()
    record = ExecutionLineageSegmentRecord(
        scope=scope,
        root_execution_id=root,
        predecessor_root_execution_id=None,
        lifecycle=ExecutionLineageSegmentLifecycle.SEGMENT_OPEN,
    )
    from intergrax.runtime.execution.lineage.codecs import (
        decode_execution_lineage_segment_record,
        encode_execution_lineage_segment_record,
    )

    payload = encode_execution_lineage_segment_record(record)
    assert decode_execution_lineage_segment_record(payload) == record


def test_codec_rejects_unknown_fields() -> None:
    scope = _scope()
    root = mint_execution_id()
    record = ExecutionLineageAdmissionRecord(
        scope=scope,
        segment_root_execution_id=root,
        execution_id=root,
        parent_execution_id=None,
        admission_position=1,
    )
    payload = encode_execution_lineage_admission_record(record)
    payload["unexpected"] = True
    with pytest.raises(Exception):
        decode_execution_lineage_admission_record(payload)


def test_scope_codec_rejects_wrong_schema_version() -> None:
    payload = encode_execution_lineage_attempt_scope(_scope())
    payload["schema_version"] = 99
    from intergrax.runtime.execution.lineage.codecs import decode_execution_lineage_attempt_scope

    with pytest.raises(Exception):
        decode_execution_lineage_attempt_scope(payload)
