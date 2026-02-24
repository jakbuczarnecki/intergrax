# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Protocol

from intergrax.runtime.replay.contracts.run_record_dto import RunRecordDTO


class RunRecordStore(Protocol):
    """Read-only access to persisted run records."""

    def get(self, tenant_id: str, run_id: str) -> RunRecordDTO:
        ...
