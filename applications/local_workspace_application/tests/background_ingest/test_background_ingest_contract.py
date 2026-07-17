# © Artur Czarnecki. All rights reserved.

"""Tests for LKW background ingest job payload contract (LKW.4A / LKW.7A)."""

from __future__ import annotations

import hashlib
import json
import re

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

_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "content",
        "chunks",
        "prompt",
        "secret",
        "document_text",
        "file_bytes",
        "embedding",
    }
)
_VALID_CHANGE_TOKEN = (
    "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)


def _legacy_background_ingest_idempotency_key(job: LkwBackgroundIngestJob) -> str:
    """Pre-LKW.7A identity: tenant/workspace/collection/source_paths only."""
    identity = {
        "tenant_id": job.tenant_id,
        "workspace_id": job.workspace_id,
        "collection_id": job.collection_id,
        "source_paths": sorted(job.source_paths),
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"lkw.background_ingest.v1:{digest[:32]}"


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
    assert decoded.change_token is None


def test_job_with_valid_change_token_roundtrips() -> None:
    job = _sample_job(change_token=_VALID_CHANGE_TOKEN)

    decoded = decode_background_ingest_job(encode_background_ingest_job(job))

    assert decoded.change_token == _VALID_CHANGE_TOKEN
    assert decoded.schema_version == LKW_BACKGROUND_INGEST_SCHEMA_VERSION


@pytest.mark.parametrize(
    "change_token",
    [
        "",
        "   ",
        "sha256:0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF0123456789ABCDEF",
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "sha256:abc",
        "sha256:" + ("a" * 65),
        "sha256:" + ("g" * 64),
    ],
)
def test_invalid_change_token_rejected(change_token: str) -> None:
    with pytest.raises(ValidationError):
        _sample_job(change_token=change_token)


def test_legacy_job_key_exactly_matches_pre_lkw7a_algorithm() -> None:
    job = _sample_job(run_id="run-ignored", correlation_id="corr-ignored", reason="x")

    assert background_ingest_idempotency_key(
        job
    ) == _legacy_background_ingest_idempotency_key(job)


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


def test_idempotency_key_ignores_reason_requested_by_and_priority() -> None:
    base = _sample_job()
    with_reason = _sample_job(reason="other-reason")
    with_requested_by = _sample_job(requested_by="other-actor")
    with_priority = _sample_job(priority="high")

    key = background_ingest_idempotency_key(base)
    assert background_ingest_idempotency_key(with_reason) == key
    assert background_ingest_idempotency_key(with_requested_by) == key
    assert background_ingest_idempotency_key(with_priority) == key


def test_same_change_token_produces_same_key() -> None:
    first = _sample_job(change_token=_VALID_CHANGE_TOKEN)
    second = _sample_job(change_token=_VALID_CHANGE_TOKEN, run_id="run-2")

    assert background_ingest_idempotency_key(
        first
    ) == background_ingest_idempotency_key(second)


def test_different_change_token_produces_different_key() -> None:
    other_token = (
        "sha256:ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )
    first = _sample_job(change_token=_VALID_CHANGE_TOKEN)
    second = _sample_job(change_token=other_token)

    assert background_ingest_idempotency_key(
        first
    ) != background_ingest_idempotency_key(second)
    assert background_ingest_idempotency_key(
        first
    ) != background_ingest_idempotency_key(_sample_job())


def test_source_path_order_remains_idempotency_stable() -> None:
    first = _sample_job(source_paths=("/data/b.txt", "/data/a.txt"))
    second = _sample_job(source_paths=("/data/a.txt", "/data/b.txt"))

    assert background_ingest_idempotency_key(
        first
    ) == background_ingest_idempotency_key(second)


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

    assert background_ingest_idempotency_key(
        changed
    ) != background_ingest_idempotency_key(base)


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
        LkwBackgroundIngestJob.model_validate(
            {
                "tenant_id": "tenant-a",
                "workspace_id": "workspace-a",
                "collection_id": "collection-a",
                "source_paths": source_paths,
            }
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
    job = _sample_job(
        reason="incremental",
        priority="high",
        change_token=_VALID_CHANGE_TOKEN,
    )
    payload = json.loads(encode_background_ingest_job(job).decode("utf-8"))

    assert _FORBIDDEN_PAYLOAD_KEYS.isdisjoint(payload.keys())
    assert job.schema_version == LKW_BACKGROUND_INGEST_SCHEMA_VERSION
    assert not re.search(
        r"\b(content|chunks|prompt|document_text|file_bytes|embedding)\b",
        json.dumps(payload),
    )


def test_background_ingest_task_name_constant() -> None:
    assert LKW_BACKGROUND_INGEST_TASK_NAME == "lkw.background_ingest.v1"
