# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""In-memory causal evidence persistence (tests, conformance, local lab)."""

from __future__ import annotations

import bisect
import secrets
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import DefaultDict

from intergrax.contracts.execution_identity import (
    RunId,
    TaskId,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.observability.causal_evidence import PlatformCausalEvidence
from intergrax.runtime.observability.causal_evidence_persistence import (
    CausalEvidencePage,
    CausalEvidencePersistence,
    CausalEvidencePersistenceConflictError,
    CausalEvidencePersistenceIntegrityError,
    causal_evidence_query_order_key,
    validate_causal_evidence_query_limit,
)
from intergrax.runtime.observability.causal_evidence_query_cursor import (
    CausalEvidenceQueryCursorCodec,
    CausalEvidenceQueryCursorError,
)

_DEFAULT_CURSOR_SECRET = secrets.token_bytes(32)


@dataclass(frozen=True, order=True, slots=True)
class _CausalEvidenceIndexEntry:
    recorded_at: datetime
    evidence_id: str


@dataclass(frozen=True, slots=True)
class _FrozenHighWater:
    recorded_at: datetime
    evidence_id: str


class InMemoryCausalEvidencePersistence(CausalEvidencePersistence):
    def __init__(
        self,
        *,
        cursor_codec: CausalEvidenceQueryCursorCodec | None = None,
        cursor_secret: bytes | None = None,
    ) -> None:
        if cursor_codec is not None and cursor_secret is not None:
            raise TypeError("causal_evidence_cursor_codec_configuration_invalid")
        if cursor_secret is not None:
            self._cursor_codec = CausalEvidenceQueryCursorCodec(secret=cursor_secret)
        else:
            self._cursor_codec = cursor_codec or CausalEvidenceQueryCursorCodec(
                secret=_DEFAULT_CURSOR_SECRET,
            )
        self._accepted_by_evidence_id: dict[str, PlatformCausalEvidence] = {}
        self._by_execution: DefaultDict[
            tuple[str, str, str], list[_CausalEvidenceIndexEntry]
        ] = defaultdict(list)
        self._by_transport: DefaultDict[
            tuple[str, str, str], list[_CausalEvidenceIndexEntry]
        ] = defaultdict(list)
        self._lock = Lock()

    @property
    def query_cursor_codec(self) -> CausalEvidenceQueryCursorCodec:
        return self._cursor_codec

    def append(self, evidence: PlatformCausalEvidence) -> PlatformCausalEvidence:
        with self._lock:
            existing = self._accepted_by_evidence_id.get(evidence.evidence_id)
            if existing is not None:
                if existing != evidence:
                    raise CausalEvidencePersistenceConflictError(
                        "conflicting causal evidence for evidence_id",
                    )
                return existing
            self._accepted_by_evidence_id[evidence.evidence_id] = evidence
            execution_key = (
                evidence.tenant_id,
                evidence.target.task_id,
                evidence.target.run_id,
            )
            transport_key = (
                evidence.tenant_id,
                evidence.source.provider,
                evidence.source.task_id,
            )
            entry = _CausalEvidenceIndexEntry(
                recorded_at=evidence.recorded_at,
                evidence_id=str(evidence.evidence_id),
            )
            bisect.insort(self._by_execution[execution_key], entry)
            bisect.insort(self._by_transport[transport_key], entry)
            return evidence

    def page_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:
        validated_limit = validate_causal_evidence_query_limit(limit)
        validated_task_id = validate_task_id(task_id)
        validated_run_id = validate_run_id(run_id)
        key = (tenant_id, validated_task_id, validated_run_id)
        with self._lock:
            return self._page_scope(
                scope_entries=self._by_execution.get(key, []),
                cursor=cursor,
                limit=validated_limit,
                decode_cursor=lambda value: self._cursor_codec.decode_execution(
                    value,
                    tenant_id=tenant_id,
                    task_id=validated_task_id,
                    run_id=validated_run_id,
                ),
                encode_cursor=lambda **kwargs: self._cursor_codec.encode_execution(
                    tenant_id=tenant_id,
                    task_id=validated_task_id,
                    run_id=validated_run_id,
                    **kwargs,
                ),
            )

    def page_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> CausalEvidencePage:
        validated_limit = validate_causal_evidence_query_limit(limit)
        key = (tenant_id, provider, transport_task_id)
        with self._lock:
            return self._page_scope(
                scope_entries=self._by_transport.get(key, []),
                cursor=cursor,
                limit=validated_limit,
                decode_cursor=lambda value: self._cursor_codec.decode_transport(
                    value,
                    tenant_id=tenant_id,
                    provider=provider,
                    transport_task_id=transport_task_id,
                ),
                encode_cursor=lambda **kwargs: self._cursor_codec.encode_transport(
                    tenant_id=tenant_id,
                    provider=provider,
                    transport_task_id=transport_task_id,
                    **kwargs,
                ),
            )

    def _page_scope(
        self,
        *,
        scope_entries: list[_CausalEvidenceIndexEntry],
        cursor: str | None,
        limit: int,
        decode_cursor,
        encode_cursor,
    ) -> CausalEvidencePage:
        if not scope_entries:
            return CausalEvidencePage(items=(), next_cursor=None)

        if cursor is None:
            snapshot_tail = scope_entries[-1]
            high_water = _FrozenHighWater(
                recorded_at=snapshot_tail.recorded_at,
                evidence_id=snapshot_tail.evidence_id,
            )
            start_pos = 0
        else:
            try:
                payload = decode_cursor(cursor)
            except CausalEvidenceQueryCursorError as exc:
                raise CausalEvidencePersistenceIntegrityError(str(exc)) from exc
            if payload.high_water is None:
                raise CausalEvidencePersistenceIntegrityError(
                    "invalid causal evidence continuation cursor",
                )
            if payload.last_recorded_at is None or payload.last_evidence_id is None:
                raise CausalEvidencePersistenceIntegrityError(
                    "invalid causal evidence continuation cursor",
                )
            high_water = _decode_high_water(payload.high_water)
            continuation = _CausalEvidenceIndexEntry(
                recorded_at=payload.last_recorded_at,
                evidence_id=payload.last_evidence_id,
            )
            start_pos = bisect.bisect_right(scope_entries, continuation)

        high_water_entry = _CausalEvidenceIndexEntry(
            recorded_at=high_water.recorded_at,
            evidence_id=high_water.evidence_id,
        )
        end_pos = bisect.bisect_right(scope_entries, high_water_entry)
        visible = scope_entries[start_pos:end_pos]
        page_entries = visible[:limit]
        items = tuple(
            self._accepted_by_evidence_id[entry.evidence_id] for entry in page_entries
        )
        has_more = len(visible) > limit
        next_cursor: str | None = None
        if has_more and page_entries:
            last_entry = page_entries[-1]
            next_cursor = encode_cursor(
                high_water=_encode_high_water(high_water),
                last_recorded_at=last_entry.recorded_at,
                last_evidence_id=last_entry.evidence_id,
                store_cursor=None,
            )
        return CausalEvidencePage(items=items, next_cursor=next_cursor)

    def list_for_execution(
        self,
        *,
        tenant_id: str,
        task_id: TaskId,
        run_id: RunId,
    ) -> tuple[PlatformCausalEvidence, ...]:
        key = (tenant_id, task_id, run_id)
        return self._resolve_entries(self._by_execution.get(key, []))

    def list_for_transport_task(
        self,
        *,
        tenant_id: str,
        provider: str,
        transport_task_id: str,
    ) -> tuple[PlatformCausalEvidence, ...]:
        key = (tenant_id, provider, transport_task_id)
        return self._resolve_entries(self._by_transport.get(key, []))

    def _resolve_entries(
        self,
        entries: list[_CausalEvidenceIndexEntry],
    ) -> tuple[PlatformCausalEvidence, ...]:
        with self._lock:
            records = [
                self._accepted_by_evidence_id[entry.evidence_id]
                for entry in entries
                if entry.evidence_id in self._accepted_by_evidence_id
            ]
            records.sort(key=causal_evidence_query_order_key)
            return tuple(records)


def _encode_high_water(high_water: _FrozenHighWater) -> str:
    return f"{high_water.recorded_at.isoformat()}|{high_water.evidence_id}"


def _decode_high_water(value: str) -> _FrozenHighWater:
    if "|" not in value:
        raise CausalEvidencePersistenceIntegrityError(
            "invalid causal evidence high-water marker"
        )
    recorded_at_raw, evidence_id = value.rsplit("|", 1)
    if not evidence_id:
        raise CausalEvidencePersistenceIntegrityError(
            "invalid causal evidence high-water marker"
        )
    try:
        recorded_at = datetime.fromisoformat(recorded_at_raw)
    except ValueError as exc:
        raise CausalEvidencePersistenceIntegrityError(
            "invalid causal evidence high-water marker",
        ) from exc
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=UTC)
    return _FrozenHighWater(recorded_at=recorded_at, evidence_id=evidence_id)
