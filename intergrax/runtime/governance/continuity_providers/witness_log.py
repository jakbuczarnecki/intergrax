# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure-shaped immutable append witness.

Blocks are fixed 4096-byte records. Append position is conditional. An unknown
append is resolved by an exact range read. Segment rollover writes one close
record and one successor genesis. A missing prefix, a corrupt block, or a
second successor fails closed. This module does not set a retention duration
and does not call Azure.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from intergrax.runtime.governance.continuity_providers.errors import ProviderFault

_BLOCK = 4096
_MAGIC = b"IGXW1\n"
_DEFAULT_SEGMENT_LIMIT = 49_000
_GENESIS_PREDECESSOR = hashlib.sha256(b"igx.offline-witness-predecessor.v1\ngenesis").hexdigest()


class AppendKind(StrEnum):
    COMMITTED = "committed"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class AppendStep:
    kind: AppendKind
    fault: ProviderFault | None = None
    segment: int | None = None
    offset: int | None = None
    digest: str | None = None


@dataclass(frozen=True, slots=True)
class WitnessView:
    fault: ProviderFault | None
    high_revision: int
    segment: int | None = None
    offset: int | None = None
    digest: str | None = None
    predecessor: str | None = None
    binding_digest: str | None = None
    fencing_token: str | None = None
    transition_id: str | None = None
    status: str | None = None
    identity_epoch: int | None = None
    tenant_id: str | None = None
    environment_id: str | None = None
    application_id: str | None = None


def commit_append(
    blob: bytes | None,
    *,
    segment: int,
    expected_offset: int,
    block: bytes,
) -> tuple[bytes | None, AppendStep]:
    """Apply one conditional append. A missing segment does not create a blob."""

    if blob is None:
        return None, AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.PREFIX_MISSING)
    if expected_offset != len(blob) or len(block) != _BLOCK:
        return blob, AppendStep(
            kind=AppendKind.CONFLICT,
            fault=ProviderFault.APPEND_POSITION_CONFLICT,
        )
    return blob + block, AppendStep(
        kind=AppendKind.COMMITTED,
        segment=segment,
        offset=expected_offset,
    )


class WitnessAppendPort(Protocol):
    """Typed witness transport. The admitted runtime owner is ``InMemoryAppendWitness``."""

    def open_segment(self, segment: int) -> None:
        """Create an empty segment when it is absent."""
        ...

    def length(self, segment: int) -> int:
        """Return the segment length, or -1 when the segment is absent."""
        ...

    def read_range(self, segment: int, offset: int, length: int) -> bytes | None:
        """Return an exact byte range, or None when the range is absent."""
        ...

    def append(self, segment: int, expected_offset: int, block: bytes) -> AppendStep:
        """Append one block when ``expected_offset`` is the current length."""
        ...


class InMemoryAppendWitness:
    """Admitted offline append transport. Conditional offset, exact read-back, one-shot faults."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segments: dict[int, bytes] = {0: b""}
        self._calls = 0
        self._fault: str | None = None
        self._fault_call = 1

    def arm_fault(self, mode: str, *, call_index: int = 1) -> None:
        if mode not in {"unknown_before", "unknown_after", "unavailable", "conflict"}:
            raise ValueError("unsupported witness fault")
        if call_index < 1:
            raise ValueError("fault call index starts at 1")
        with self._lock:
            self._fault = mode
            self._fault_call = call_index
            self._calls = 0

    def clear_fault(self) -> None:
        with self._lock:
            self._fault = None
            self._calls = 0

    def open_segment(self, segment: int) -> None:
        with self._lock:
            self._segments.setdefault(segment, b"")

    def length(self, segment: int) -> int:
        with self._lock:
            blob = self._segments.get(segment)
            if blob is None:
                return -1
            return len(blob)

    def read_range(self, segment: int, offset: int, length: int) -> bytes | None:
        with self._lock:
            blob = self._segments.get(segment)
            if blob is None or offset < 0 or length < 0 or offset + length > len(blob):
                return None
            return blob[offset : offset + length]

    def append(self, segment: int, expected_offset: int, block: bytes) -> AppendStep:
        with self._lock:
            self._calls += 1
            blob = self._segments.get(segment)
            fault = self._fault if self._calls == self._fault_call else None
            if fault is not None:
                self._fault = None
            if blob is None:
                return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.PREFIX_MISSING)
            if fault == "unavailable":
                return AppendStep(kind=AppendKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)
            if fault == "unknown_before":
                return AppendStep(kind=AppendKind.UNKNOWN, fault=ProviderFault.OUTCOME_UNKNOWN)
            if fault == "conflict":
                return AppendStep(
                    kind=AppendKind.CONFLICT,
                    fault=ProviderFault.APPEND_POSITION_CONFLICT,
                )
            updated, step = commit_append(
                blob,
                segment=segment,
                expected_offset=expected_offset,
                block=block,
            )
            if step.kind is not AppendKind.COMMITTED or updated is None:
                return step
            self._segments[segment] = updated
            if fault == "unknown_after":
                return AppendStep(
                    kind=AppendKind.UNKNOWN,
                    fault=ProviderFault.OUTCOME_UNKNOWN,
                    segment=segment,
                    offset=expected_offset,
                )
            return step

    def testing_truncate(self, segment: int, new_length: int) -> None:
        with self._lock:
            blob = self._segments.get(segment)
            if blob is None or new_length < 0 or new_length > len(blob):
                raise RuntimeError("witness truncate is outside the segment")
            self._segments[segment] = blob[:new_length]

    def testing_cut_prefix(self, segment: int, nbytes: int) -> None:
        with self._lock:
            blob = self._segments.get(segment)
            if blob is None or nbytes < 1 or nbytes > len(blob):
                raise RuntimeError("witness prefix cut is outside the segment")
            self._segments[segment] = blob[nbytes:]

    def testing_replace_block(self, segment: int, index: int, block: bytes) -> None:
        if len(block) != _BLOCK:
            raise RuntimeError("witness block must be 4096 bytes")
        with self._lock:
            blob = self._segments.get(segment)
            start = index * _BLOCK
            if blob is None or start + _BLOCK > len(blob):
                raise RuntimeError("witness block is absent")
            self._segments[segment] = blob[:start] + block + blob[start + _BLOCK :]


def encode_block(payload: dict[str, object]) -> bytes:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    raw = _MAGIC + len(body).to_bytes(4, "big") + body
    if len(raw) > _BLOCK:
        raise ValueError("witness block exceeds 4096 bytes")
    return raw + bytes(_BLOCK - len(raw))


def decode_block(block: bytes) -> dict[str, object] | None:
    if len(block) != _BLOCK or not block.startswith(_MAGIC):
        return None
    size = int.from_bytes(block[len(_MAGIC) : len(_MAGIC) + 4], "big")
    start = len(_MAGIC) + 4
    end = start + size
    if size < 2 or end > _BLOCK:
        return None
    try:
        payload = json.loads(block[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def block_digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class AzureAppendWitness:
    """Conditional append, digest chain, rollover, and bounded verification."""

    def __init__(
        self,
        store: WitnessAppendPort,
        *,
        segment_block_limit: int = _DEFAULT_SEGMENT_LIMIT,
        recovery_block_bound: int = 128,
    ) -> None:
        if segment_block_limit < 3:
            raise ValueError("a segment must fit a seal, a close, and a later genesis")
        if recovery_block_bound < 1:
            raise ValueError("recovery bound must be positive")
        self._store = store
        self._limit = segment_block_limit
        self._bound = recovery_block_bound

    @property
    def segment_block_limit(self) -> int:
        return self._limit

    def append_seal(
        self,
        *,
        application_id: str,
        environment_id: str,
        tenant_id: str,
        revision: int,
        fencing_token: str,
        transition_id: str,
        binding_digest: str,
        status: str,
        identity_epoch: int,
    ) -> AppendStep:
        tail = self._tail_for_append()
        if tail.fault is not None:
            return AppendStep(kind=AppendKind.CONFLICT, fault=tail.fault)
        if tail.high_revision and (
            tail.application_id != application_id
            or tail.environment_id != environment_id
            or tail.tenant_id != tenant_id
        ):
            return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.SCOPE_MISMATCH)
        if revision != tail.high_revision + 1:
            return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.DIGEST_MISMATCH)
        if self._block_count() >= self._bound:
            return AppendStep(kind=AppendKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)
        if tail.high_revision == 0:
            predecessor = _GENESIS_PREDECESSOR
        elif tail.digest is None:
            return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.CORRUPT)
        else:
            predecessor = tail.digest
        payload = _seal_payload(
            application_id=application_id,
            environment_id=environment_id,
            tenant_id=tenant_id,
            revision=revision,
            fencing_token=fencing_token,
            transition_id=transition_id,
            binding_digest=binding_digest,
            status=status,
            identity_epoch=identity_epoch,
            predecessor=predecessor,
        )
        slot = self._slot_for_seal()
        if isinstance(slot, AppendStep):
            return slot
        segment, ordinal = slot
        return self._append_payload(segment, ordinal, payload)

    def _tail_for_append(self) -> WitnessView:
        """Tail used to extend a chain. An empty successor is not yet a fault."""

        located = self._live_segment()
        if isinstance(located, WitnessView):
            return located
        length = self._store.length(located)
        if located > 0 and length == 0:
            return self._last_seal_before(located)
        if length >= _BLOCK:
            last = self._block(located, (length // _BLOCK) - 1)
            if last is not None and last[1].get("kind") == "segment_genesis":
                return self._last_seal_before(located)
        return self.verify_bounded(full_chain=False)

    def _last_seal_before(self, segment: int) -> WitnessView:
        previous = self._previous_seal(segment, 0)
        if previous is None:
            return WitnessView(fault=ProviderFault.PREFIX_MISSING, high_revision=0)
        revision = previous.get("revision")
        if type(revision) is not int:
            return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
        return self._view_from_seal(segment - 1, 0, previous)

    def verify_bounded(self, *, full_chain: bool) -> WitnessView:
        if self._store.length(0) < 0:
            return WitnessView(fault=ProviderFault.PREFIX_MISSING, high_revision=0)
        if full_chain:
            return self._verify_full_chain()
        return self._verify_tail()

    def _verify_tail(self) -> WitnessView:
        origin = self._block(0, 0)
        if origin is None:
            return WitnessView(fault=None, high_revision=0, predecessor=_GENESIS_PREDECESSOR)
        if origin[1].get("kind") != "seal" or origin[1].get("predecessor") != _GENESIS_PREDECESSOR:
            return WitnessView(fault=ProviderFault.PREFIX_MISSING, high_revision=0)
        if origin[1].get("revision") != 1:
            return WitnessView(fault=ProviderFault.PREFIX_MISSING, high_revision=0)
        segment = self._live_segment()
        if isinstance(segment, WitnessView):
            return segment
        length = self._store.length(segment)
        if length % _BLOCK != 0:
            return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
        count = length // _BLOCK
        if count == 0:
            return WitnessView(fault=ProviderFault.UNAVAILABLE, high_revision=0)
        raw, payload = self._block(segment, count - 1) or (b"", {})
        if not payload:
            return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
        if payload.get("kind") == "segment_genesis":
            return WitnessView(fault=ProviderFault.UNAVAILABLE, high_revision=0)
        if payload.get("kind") != "seal":
            return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
        linked = self._predecessor_matches(segment, count - 1, payload)
        if linked is not None:
            return WitnessView(fault=linked, high_revision=0)
        return self._view_from_seal(segment, count - 1, payload)

    def _predecessor_matches(
        self,
        segment: int,
        index: int,
        payload: dict[str, object],
    ) -> ProviderFault | None:
        if payload.get("revision") == 1:
            if payload.get("predecessor") != _GENESIS_PREDECESSOR:
                return ProviderFault.PREFIX_MISSING
            return None
        previous = self._previous_seal(segment, index)
        if previous is None:
            return ProviderFault.PREFIX_MISSING
        if payload.get("predecessor") != block_digest(previous):
            return ProviderFault.CORRUPT
        return None

    def _previous_seal(self, segment: int, index: int) -> dict[str, object] | None:
        cursor_segment = segment
        cursor_index = index - 1
        while cursor_segment >= 0:
            if cursor_index < 0:
                if cursor_segment == 0:
                    return None
                cursor_segment -= 1
                length = self._store.length(cursor_segment)
                if length < _BLOCK:
                    return None
                cursor_index = (length // _BLOCK) - 1
            found = self._block(cursor_segment, cursor_index)
            if found is None:
                return None
            kind = found[1].get("kind")
            if kind == "seal":
                return found[1]
            cursor_index -= 1
        return None

    def _verify_full_chain(self) -> WitnessView:
        expected_predecessor = _GENESIS_PREDECESSOR
        high = 0
        last = WitnessView(fault=None, high_revision=0, predecessor=_GENESIS_PREDECESSOR)
        seen = 0
        segment = 0
        while True:
            length = self._store.length(segment)
            if length < 0:
                return WitnessView(fault=ProviderFault.PREFIX_MISSING, high_revision=high)
            if length % _BLOCK != 0:
                return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
            count = length // _BLOCK
            if count == 0:
                if segment == 0:
                    return last
                return WitnessView(fault=ProviderFault.UNAVAILABLE, high_revision=high)
            for index in range(count):
                seen += 1
                if seen > self._bound:
                    return WitnessView(fault=ProviderFault.UNAVAILABLE, high_revision=high)
                found = self._block(segment, index)
                if found is None:
                    return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
                payload = found[1]
                kind = payload.get("kind")
                if kind == "segment_closed":
                    if index != count - 1 or payload.get("successor_segment") != segment + 1:
                        return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
                    prior = self._store.read_range(segment, 0, index * _BLOCK) or b""
                    if payload.get("predecessor") != hashlib.sha256(prior).hexdigest():
                        return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
                    continue
                if kind == "segment_genesis":
                    if index != 0 or segment == 0:
                        return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
                    continue
                if kind != "seal":
                    return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
                if payload.get("predecessor") != expected_predecessor:
                    fault = ProviderFault.PREFIX_MISSING if high == 0 else ProviderFault.CORRUPT
                    return WitnessView(fault=fault, high_revision=high)
                revision = payload.get("revision")
                if type(revision) is not int or revision != high + 1:
                    return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
                high = revision
                expected_predecessor = block_digest(payload)
                last = self._view_from_seal(segment, index, payload)
            tail = self._block(segment, count - 1)
            if tail is None:
                return WitnessView(fault=ProviderFault.CORRUPT, high_revision=high)
            if tail[1].get("kind") != "segment_closed":
                return last
            segment += 1

    def _live_segment(self) -> int | WitnessView:
        segment = 0
        hops = 0
        while hops <= self._bound:
            hops += 1
            length = self._store.length(segment)
            if length < 0:
                return WitnessView(fault=ProviderFault.PREFIX_MISSING, high_revision=0)
            if length % _BLOCK != 0:
                return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
            count = length // _BLOCK
            if count == 0:
                return segment
            tail = self._block(segment, count - 1)
            if tail is None:
                return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
            if tail[1].get("kind") != "segment_closed":
                return segment
            if tail[1].get("successor_segment") != segment + 1:
                return WitnessView(fault=ProviderFault.CORRUPT, high_revision=0)
            if self._store.length(segment + 1) < 0:
                return WitnessView(fault=ProviderFault.UNAVAILABLE, high_revision=0)
            segment += 1
        return WitnessView(fault=ProviderFault.UNAVAILABLE, high_revision=0)

    def _slot_for_seal(self) -> tuple[int, int] | AppendStep:
        located = self._live_segment()
        if isinstance(located, WitnessView):
            return AppendStep(
                kind=AppendKind.CONFLICT,
                fault=located.fault or ProviderFault.UNAVAILABLE,
            )
        segment = located
        length = self._store.length(segment)
        ordinal = length // _BLOCK
        if segment > 0 and ordinal == 0:
            opened = self._append_payload(segment, 0, self._genesis_payload(segment - 1))
            if opened.kind is not AppendKind.COMMITTED:
                return opened
            ordinal = 1
        if ordinal < self._limit - 1:
            return (segment, ordinal)
        if ordinal > self._limit - 1:
            return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.CORRUPT)
        closed = self._close_and_open(segment)
        if closed.kind is not AppendKind.COMMITTED:
            return closed
        return (segment + 1, self._store.length(segment + 1) // _BLOCK)

    def _block(self, segment: int, index: int) -> tuple[bytes, dict[str, object]] | None:
        raw = self._store.read_range(segment, index * _BLOCK, _BLOCK)
        if raw is None:
            return None
        payload = decode_block(raw)
        if payload is None:
            return None
        return (raw, payload)

    def _view_from_seal(self, segment: int, index: int, payload: dict[str, object]) -> WitnessView:
        revision = payload.get("revision")
        epoch = payload.get("identity_epoch")
        return WitnessView(
            fault=None,
            high_revision=revision if type(revision) is int else 0,
            segment=segment,
            offset=index * _BLOCK,
            digest=block_digest(payload),
            predecessor=str(payload.get("predecessor")),
            binding_digest=str(payload.get("binding_digest")),
            fencing_token=str(payload.get("fencing_token")),
            transition_id=str(payload.get("transition_id")),
            status=str(payload.get("status")),
            identity_epoch=epoch if type(epoch) is int else None,
            tenant_id=str(payload.get("tenant_id")),
            environment_id=str(payload.get("environment_id")),
            application_id=str(payload.get("application_id")),
        )

    def _block_count(self) -> int:
        total = 0
        segment = 0
        while total <= self._bound:
            length = self._store.length(segment)
            if length < 0:
                return total
            if length % _BLOCK != 0:
                return self._bound + 1
            total += length // _BLOCK
            segment += 1
        return total

    def _genesis_payload(self, closed_segment: int) -> dict[str, object]:
        length = self._store.length(closed_segment)
        found = self._block(closed_segment, (length // _BLOCK) - 1)
        predecessor = ""
        if found is not None and found[1].get("kind") == "segment_closed":
            stored = found[1].get("predecessor")
            if type(stored) is str:
                predecessor = stored
        return {
            "kind": "segment_genesis",
            "predecessor_segment_digest": predecessor,
        }

    def _close_and_open(self, segment: int) -> AppendStep:
        close_payload: dict[str, object] = {
            "kind": "segment_closed",
            "segment": segment,
            "successor_segment": segment + 1,
            "predecessor": _segment_digest(self._store, segment),
        }
        closed = self._append_payload(segment, self._limit - 1, close_payload)
        if closed.kind is not AppendKind.COMMITTED:
            return closed
        self._store.open_segment(segment + 1)
        genesis: dict[str, object] = {
            "kind": "segment_genesis",
            "segment": segment + 1,
            "predecessor_segment_digest": close_payload["predecessor"],
        }
        opened = self._append_payload(segment + 1, 0, genesis)
        return opened

    def _append_payload(self, segment: int, ordinal: int, payload: dict[str, object]) -> AppendStep:
        payload = dict(payload)
        payload["segment"] = segment
        payload["record_ordinal"] = ordinal
        block = encode_block(payload)
        digest = block_digest(payload)
        offset = ordinal * _BLOCK
        existing = self._store.read_range(segment, offset, _BLOCK)
        if existing is not None:
            if existing == block:
                return AppendStep(
                    kind=AppendKind.COMMITTED,
                    segment=segment,
                    offset=offset,
                    digest=digest,
                )
            return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.CORRUPT)
        step = self._store.append(segment, offset, block)
        if step.kind is AppendKind.COMMITTED:
            return AppendStep(kind=AppendKind.COMMITTED, segment=segment, offset=offset, digest=digest)
        if step.kind is AppendKind.UNKNOWN:
            read_back = self._store.read_range(segment, offset, _BLOCK)
            if read_back == block:
                return AppendStep(
                    kind=AppendKind.COMMITTED,
                    segment=segment,
                    offset=offset,
                    digest=digest,
                )
            if read_back is None:
                return AppendStep(kind=AppendKind.UNKNOWN, fault=ProviderFault.OUTCOME_UNKNOWN)
            return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.CORRUPT)
        if step.kind is AppendKind.CONFLICT:
            read_back = self._store.read_range(segment, offset, _BLOCK)
            if read_back == block:
                return AppendStep(
                    kind=AppendKind.COMMITTED,
                    segment=segment,
                    offset=offset,
                    digest=digest,
                )
            return AppendStep(
                kind=AppendKind.CONFLICT,
                fault=step.fault or ProviderFault.APPEND_POSITION_CONFLICT,
            )
        return AppendStep(kind=AppendKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)


def _segment_digest(store: WitnessAppendPort, segment: int) -> str:
    length = store.length(segment)
    raw = store.read_range(segment, 0, length) if length > 0 else b""
    return hashlib.sha256(raw or b"").hexdigest()


def _seal_payload(
    *,
    application_id: str,
    environment_id: str,
    tenant_id: str,
    revision: int,
    fencing_token: str,
    transition_id: str,
    binding_digest: str,
    status: str,
    identity_epoch: int,
    predecessor: str,
) -> dict[str, object]:
    return {
        "kind": "seal",
        "application_id": application_id,
        "environment_id": environment_id,
        "tenant_id": tenant_id,
        "revision": revision,
        "fencing_token": fencing_token,
        "transition_id": transition_id,
        "binding_digest": binding_digest,
        "status": status,
        "identity_epoch": identity_epoch,
        "predecessor": predecessor,
    }
