# © Artur Czarnecki. All rights reserved.

"""Tests for LKW background ingest job payload contract (LKW.4A)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from intergrax.tools.providers.message_bus.contracts import MessageBusEnqueueInput
from local_workspace_application.background_ingest.contracts import (
    LKW_BACKGROUND_INGEST_SCHEMA_VERSION,
    LKW_BACKGROUND_INGEST_TASK_NAME,
    LkwBackgroundIngestJob,
    background_ingest_idempotency_key,
    background_ingest_payload_base64,
    decode_background_ingest_job,
    encode_background_ingest_job,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_PAYLOAD_KEYS = frozenset({"content", "chunks", "prompt", "secret"})


def _sample_job(**overrides: object) -> LkwBackgroundIngestJob:
    payload = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "collection_id": "collection-a",
        "source_paths": ("/data/user_docs/proof.txt",),
    }
    payload.update(overrides)
    return LkwBackgroundIngestJob.model_validate(payload)


def test_background_ingest_job_encodes_and_decodes_roundtrip() -> None:
    job = _sample_job(run_id="run-1", correlation_id="corr-1", reason="watcher-batch")

    encoded = encode_background_ingest_job(job)
    decoded = decode_background_ingest_job(encoded)

    assert decoded == job


def test_encoded_bytes_are_deterministic_for_equivalent_models() -> None:
    first = _sample_job(source_paths=[" /data/a.txt ", "/data/b.txt"])
    second = _sample_job(source_paths=("/data/a.txt", "/data/b.txt"))

    assert encode_background_ingest_job(first) == encode_background_ingest_job(second)


def test_idempotency_key_is_stable_across_run_and_correlation_changes() -> None:
    base = _sample_job()
    with_run = _sample_job(run_id="run-1")
    with_correlation = _sample_job(correlation_id="corr-1")
    with_both = _sample_job(run_id="run-2", correlation_id="corr-2")

    key = background_ingest_idempotency_key(base)
    assert background_ingest_idempotency_key(with_run) == key
    assert background_ingest_idempotency_key(with_correlation) == key
    assert background_ingest_idempotency_key(with_both) == key


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "tenant-b"),
        ("workspace_id", "workspace-b"),
        ("collection_id", "collection-b"),
        ("source_paths", ("/data/other.txt",)),
    ],
)
def test_idempotency_key_changes_when_identity_fields_change(
    field: str,
    value: object,
) -> None:
    base = _sample_job()
    changed = _sample_job(**{field: value})

    assert background_ingest_idempotency_key(changed) != background_ingest_idempotency_key(base)


def test_source_paths_are_stripped() -> None:
    job = _sample_job(source_paths=["  /data/a.txt  ", " /data/b.txt"])

    assert job.source_paths == ("/data/a.txt", "/data/b.txt")


@pytest.mark.parametrize(
    "source_paths",
    [
        [""],
        ["   "],
        ["/data/a.txt", ""],
        [],
    ],
)
def test_blank_source_paths_are_rejected(source_paths: list[str]) -> None:
    with pytest.raises(ValidationError):
        LkwBackgroundIngestJob(
            tenant_id="tenant-a",
            workspace_id="workspace-a",
            collection_id="collection-a",
            source_paths=source_paths,
        )


def test_payload_base64_is_accepted_by_message_bus_enqueue_validation() -> None:
    job = _sample_job(run_id="run-1", correlation_id="corr-1")
    payload_base64 = background_ingest_payload_base64(job)

    validated = MessageBusEnqueueInput(
        tenant_id=job.tenant_id,
        run_id="enqueue-run",
        task_name=LKW_BACKGROUND_INGEST_TASK_NAME,
        payload_base64=payload_base64,
        idempotency_key=background_ingest_idempotency_key(job),
    )

    assert validated.payload_base64 == payload_base64


def test_encoded_payload_excludes_raw_content_like_fields() -> None:
    job = _sample_job(reason="incremental", priority="high")
    payload = json.loads(encode_background_ingest_job(job).decode("utf-8"))

    assert _FORBIDDEN_PAYLOAD_KEYS.isdisjoint(payload.keys())
    assert job.schema_version == LKW_BACKGROUND_INGEST_SCHEMA_VERSION


def test_background_ingest_task_name_constant() -> None:
    assert LKW_BACKGROUND_INGEST_TASK_NAME == "lkw.background_ingest.v1"
