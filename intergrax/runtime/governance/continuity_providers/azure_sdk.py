# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Azure Blob append-client boundary.

Construction refuses before importing or calling the client. Protected append
and the combined hold are not applied. No retention duration is set. This
module does not qualify Azure.
"""

from __future__ import annotations

import importlib

from intergrax.runtime.governance.continuity_providers.credentials import (
    OfflineProviderRefusal,
    evaluate_continuity_credentials,
)
from intergrax.runtime.governance.continuity_providers.targets import AzureWitnessTarget

_CLOUD_EXECUTION_AUTHORIZED = False


def cloud_execution_authorized() -> bool:
    return _CLOUD_EXECUTION_AUTHORIZED


def open_azure_append_client(target: AzureWitnessTarget, binding: object) -> object:
    """Refuse a live append client. The SDK import below is not reached."""

    decision = evaluate_continuity_credentials(binding)
    if (
        not _CLOUD_EXECUTION_AUTHORIZED
        or decision.token is None
        or type(target) is not AzureWitnessTarget
        or target.immutability_period_days is not None
        or target.protected_append != "unqualified"
        or target.combined_hold != "unqualified"
    ):
        raise OfflineProviderRefusal(decision.refused.value)
    blobs = importlib.import_module("azure.storage.blob")
    blob = getattr(blobs, "BlobClient").from_blob_url("https://offline.invalid/witness")

    def _append(data: bytes, expected_offset: int):
        return blob.append_block(data, length=len(data), appendpos_condition=expected_offset)

    def _read(offset: int, length: int) -> bytes:
        return blob.download_blob(offset=offset, length=length).readall()

    return _append, _read
