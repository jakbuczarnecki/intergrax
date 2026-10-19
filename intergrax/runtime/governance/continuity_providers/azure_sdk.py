# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure append-witness transport.

``open_azure_append_client`` always refuses a live session. It has no switch
that can import or call a vendor client. ``AzureAppendTransport`` executes the
documented append-blob argument shapes locally. ``AzureSdkAppendBridge``
translates those same shapes onto an injected blob directory. Constructing the
bridge does not open a client, acquire a credential, or contact a network.
Protected append and the combined hold are not applied. Neither path qualifies
the vendor.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NoReturn, Protocol

from intergrax.contracts.governance.history_continuity import HistoryContinuityIdentityBinding
from intergrax.runtime.governance.continuity_providers.credentials import (
    OfflineProviderRefusal,
    evaluate_continuity_credentials,
)
from intergrax.runtime.governance.continuity_providers.errors import (
    ProviderFault,
    map_transport_exception,
)
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


_SECONDARY_HOST_MARK = "-secondary.blob."


class WitnessStorageUnavailable(Exception):
    """Storage could not answer. A missing blob is a separate result."""

    def __init__(self, fault: ProviderFault) -> None:
        super().__init__(fault.value)
        self.fault = fault


class AppendBlobDownload(Protocol):
    """Body returned by ``download_blob`` before ``readall``."""

    def readall(self) -> bytes:
        """Return the requested bytes. A short body is not a full range."""
        ...


class AppendBlobProperties(Protocol):
    """Properties returned by ``get_blob_properties``. ``size`` is the length."""

    @property
    def size(self) -> int:
        """Committed byte length of the append blob."""
        ...


class AppendBlobSession(Protocol):
    """Injected append-blob client for one named blob. Not a session factory."""

    @property
    def url(self) -> str:
        """Blob URL. A secondary host is not a positive freshness source."""
        ...

    def create_append_blob(self) -> Mapping[str, str | int]:
        """Create one empty append blob. No retention, hold, or immutability."""
        ...

    def append_block(
        self,
        data: bytes,
        length: int | None = None,
        *,
        appendpos_condition: int,
    ) -> Mapping[str, str | int]:
        """Append ``data`` only when the current length equals ``appendpos_condition``."""
        ...

    def download_blob(
        self,
        offset: int | None = None,
        length: int | None = None,
    ) -> AppendBlobDownload:
        """Download one exact range from the primary host."""
        ...

    def get_blob_properties(self) -> AppendBlobProperties:
        """Read the primary length. This does not download a body."""
        ...


class AppendBlobDirectory(Protocol):
    """Injected container handle. ``get_blob_client`` does not send a request."""

    def get_blob_client(self, blob: str) -> AppendBlobSession:
        """Return the client for one deterministic witness blob name."""
        ...


def _exception_status(exc: BaseException) -> int | None:
    status = exc.__dict__.get("status_code")
    if type(status) is int:
        return status
    return None


def _blob_is_missing(exc: BaseException) -> bool:
    if type(exc).__name__ == "ResourceNotFoundError":
        return True
    return _exception_status(exc) == 404


def _blob_already_exists(exc: BaseException) -> bool:
    if type(exc).__name__ == "ResourceExistsError":
        return True
    return _exception_status(exc) == 409


def _positive_read_allowed(blob: AppendBlobSession) -> bool:
    return _SECONDARY_HOST_MARK not in blob.url


class AzureSdkAppendBridge:
    """Witness port over an injected blob directory. The hash chain stays in the witness."""

    def __init__(self, directory: AppendBlobDirectory) -> None:
        self._directory = directory
        self._lock = threading.RLock()

    def open_segment(self, segment: int) -> None:
        with self._lock:
            blob = self._blob(segment)
            if not _positive_read_allowed(blob):
                raise WitnessStorageUnavailable(ProviderFault.UNAVAILABLE)
            try:
                blob.get_blob_properties()
            except Exception as exc:
                if not _blob_is_missing(exc):
                    raise WitnessStorageUnavailable(map_transport_exception(exc)) from exc
            else:
                return
            try:
                blob.create_append_blob()
            except Exception as exc:
                if _blob_already_exists(exc):
                    return
                raise WitnessStorageUnavailable(map_transport_exception(exc)) from exc

    def length(self, segment: int) -> int:
        with self._lock:
            blob = self._blob(segment)
            if not _positive_read_allowed(blob):
                return -1
            try:
                return blob.get_blob_properties().size
            except Exception as exc:
                if _blob_is_missing(exc):
                    return -1
                raise WitnessStorageUnavailable(map_transport_exception(exc)) from exc

    def read_range(self, segment: int, offset: int, length: int) -> bytes | None:
        with self._lock:
            if offset < 0 or length < 0:
                return None
            blob = self._blob(segment)
            if not _positive_read_allowed(blob):
                return None
            try:
                payload = blob.download_blob(offset=offset, length=length).readall()
            except Exception as exc:
                if _blob_is_missing(exc):
                    return None
                raise WitnessStorageUnavailable(map_transport_exception(exc)) from exc
            if len(payload) != length:
                return None
            return payload

    def append(self, segment: int, expected_offset: int, block: bytes) -> AppendStep:
        with self._lock:
            blob = self._blob(segment)
            if not _positive_read_allowed(blob):
                return AppendStep(kind=AppendKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)
            try:
                blob.append_block(block, length=len(block), appendpos_condition=expected_offset)
            except Exception as exc:
                if _blob_is_missing(exc):
                    return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.PREFIX_MISSING)
                fault = map_transport_exception(exc)
                if fault is ProviderFault.APPEND_POSITION_CONFLICT:
                    return AppendStep(kind=AppendKind.CONFLICT, fault=fault)
                if fault is ProviderFault.OUTCOME_UNKNOWN:
                    return self._resolve_uncertain_append(blob, segment, expected_offset, block)
                return AppendStep(kind=AppendKind.UNAVAILABLE, fault=ProviderFault.UNAVAILABLE)
            return AppendStep(
                kind=AppendKind.COMMITTED,
                segment=segment,
                offset=expected_offset,
            )

    def _blob(self, segment: int) -> AppendBlobSession:
        return self._directory.get_blob_client(witness_blob_name(segment))

    def _resolve_uncertain_append(
        self,
        blob: AppendBlobSession,
        segment: int,
        expected_offset: int,
        block: bytes,
    ) -> AppendStep:
        unknown = AppendStep(kind=AppendKind.UNKNOWN, fault=ProviderFault.OUTCOME_UNKNOWN)
        try:
            payload = blob.download_blob(offset=expected_offset, length=len(block)).readall()
        except Exception:
            return unknown
        if payload == block:
            return AppendStep(
                kind=AppendKind.COMMITTED,
                segment=segment,
                offset=expected_offset,
            )
        if len(payload) != len(block):
            return unknown
        return AppendStep(kind=AppendKind.CONFLICT, fault=ProviderFault.CORRUPT)
