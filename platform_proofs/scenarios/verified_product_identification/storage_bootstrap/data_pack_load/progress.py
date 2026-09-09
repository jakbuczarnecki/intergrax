"""Optional progress reporting for Data Pack storage bootstrap."""

from __future__ import annotations

from typing import Protocol

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapProgress,
)


class BootstrapProgressSinkPort(Protocol):
    def emit(self, progress: BootstrapProgress) -> None: ...
