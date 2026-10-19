# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure append-witness transport.

``open_azure_append_client`` always refuses a live session. It has no switch
that can import or call the Azure client. ``AzureAppendTransport`` executes the
documented ``BlobClient.create_append_blob``, ``append_block``,
``download_blob``, and ``get_blob_properties`` argument shapes locally.
Protected append and the combined hold are not applied. This is offline code
evidence. It is not an Azure session and it does not qualify the vendor.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import NoReturn

from intergrax.contracts.governance.history_continuity import HistoryContinuityIdentityBinding
from intergrax.runtime.governance.continuity_providers.credentials import (
    OfflineProviderRefusal,
    evaluate_continuity_credentials,
)
from intergrax.runtime.governance.continuity_providers.errors import ProviderFault
from intergrax.runtime.governance.continuity_providers.targets import AzureWitnessTarget
from intergrax.runtime.governance.continuity_providers.witness_log import (
    AppendKind,
    AppendStep,
    commit_append,
)

_BLOB_PREFIX = "continuity-witness/segment-"
_BLOB_SUFFIX = ".bin"


@dataclass(frozen=True, slots=True)
class CreateAppendBlob:
    """``BlobClient.create_append_blob`` for one named append blob. No client."""

    blob_name: str


@dataclass(frozen=True, slots=True)
class AppendBlock:
    """Arguments of ``BlobClient.append_block``. ``data`` is the block body."""

    blob_name: str
    data: bytes
    length: int
    appendpos_condition: int


@dataclass(frozen=True, slots=True)
class DownloadBlob:
    """Arguments of ``BlobClient.download_blob`` before ``readall``. No client."""

    blob_name: str
    offset: int
    length: int


@dataclass(frozen=True, slots=True)
class GetBlobProperties:
    """``BlobClient.get_blob_properties``; the offline result is the byte length."""

    blob_name: str


AzureWitnessCall = CreateAppendBlob | AppendBlock | DownloadBlob | GetBlobProperties


def cloud_execution_authorized() -> bool:
    return False


def witness_blob_name(segment: int) -> str:
    return f"{_BLOB_PREFIX}{segment}{_BLOB_SUFFIX}"


def segment_from_blob_name(blob_name: str) -> int:
    if not blob_name.startswith(_BLOB_PREFIX) or not blob_name.endswith(_BLOB_SUFFIX):
        raise ValueError("witness blob name is outside the provider layout")
    raw = blob_name[len(_BLOB_PREFIX) : -len(_BLOB_SUFFIX)]
    if raw == "" or not raw.isascii() or not raw.isdigit():
        raise ValueError("witness blob name is outside the provider layout")
    return int(raw)


def open_azure_append_client(
    target: AzureWitnessTarget,
    binding: HistoryContinuityIdentityBinding | None,
) -> NoReturn:
    """Refuse a live append client before any SDK import or network call."""

    decision = evaluate_continuity_credentials(binding)
    if type(target) is not AzureWitnessTarget:
        raise OfflineProviderRefusal(decision.refused.value)
    if target.immutability_period_days is not None:
        raise OfflineProviderRefusal(decision.refused.value)
    if target.protected_append != "unqualified" or target.combined_hold != "unqualified":
        raise OfflineProviderRefusal(decision.refused.value)
    raise OfflineProviderRefusal(decision.refused.value)


def replay_witness_calls(calls: tuple[AzureWitnessCall, ...]) -> dict[int, bytes]:
    """Replay create and append calls. Reads do not change the blobs."""

    segments: dict[int, bytes] = {}
    for call in calls:
        if type(call) is CreateAppendBlob:
            segments.setdefault(segment_from_blob_name(call.blob_name), b"")
            continue
        if type(call) is DownloadBlob or type(call) is GetBlobProperties:
            continue
        if type(call) is not AppendBlock:
            raise ValueError("unsupported witness call")
        if call.length != len(call.data):
            raise ValueError("append length does not match the block")
        segment = segment_from_blob_name(call.blob_name)
        updated, step = commit_append(
            segments.get(segment),
            segment=segment,
            expected_offset=call.appendpos_condition,
            block=call.data,
        )
        if step.kind is not AppendKind.COMMITTED or updated is None:
            continue
        segments[segment] = updated
    return segments


class AzureAppendTransport:
    """Offline executor for the Azure append-blob call shapes. Not a live client."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._segments: dict[int, bytes] = {}
        self.operations: list[AzureWitnessCall] = []
        self.open_segment(0)

    def open_segment(self, segment: int) -> None:
        with self._lock:
            self.operations.append(CreateAppendBlob(witness_blob_name(segment)))
            self._segments.setdefault(segment, b"")
            self._require_replay()

    def length(self, segment: int) -> int:
        with self._lock:
            self.operations.append(GetBlobProperties(witness_blob_name(segment)))
            blob = self._segments.get(segment)
            if blob is None:
                return -1
            return len(blob)

    def read_range(self, segment: int, offset: int, length: int) -> bytes | None:
        with self._lock:
            self.operations.append(
                DownloadBlob(
                    blob_name=witness_blob_name(segment),
                    offset=offset,
                    length=length,
                )
            )
            blob = self._segments.get(segment)
            if blob is None or offset < 0 or length < 0 or offset + length > len(blob):
                return None
            return blob[offset : offset + length]

    def append(self, segment: int, expected_offset: int, block: bytes) -> AppendStep:
        with self._lock:
            previous = self._segments.get(segment)
            self.operations.append(
                AppendBlock(
                    blob_name=witness_blob_name(segment),
                    data=block,
                    length=len(block),
                    appendpos_condition=expected_offset,
                )
            )
            updated, step = commit_append(
                previous,
                segment=segment,
                expected_offset=expected_offset,
                block=block,
            )
            if step.kind is AppendKind.COMMITTED and updated is not None:
                self._segments[segment] = updated
            try:
                self._require_replay()
            except ValueError:
                self.operations.pop()
                if previous is None:
                    self._segments.pop(segment, None)
                else:
                    self._segments[segment] = previous
                return AppendStep(kind=AppendKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)
            return step

    def _require_replay(self) -> None:
        if replay_witness_calls(tuple(self.operations)) != self._segments:
            raise ValueError("witness call log diverges from the blob image")
