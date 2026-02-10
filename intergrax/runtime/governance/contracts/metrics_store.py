# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from typing import Protocol, List

from intergrax.runtime.governance.contracts.metrics_record_dto import RunMetricsRecord


class ExecutionMetricsStore(Protocol):
    """
    Storage contract for behavioral metrics history.
    """

    def save(self, record: RunMetricsRecord) -> None: ...

    def get_recent(
        self,
        agent_id: str,
        limit: int,
    ) -> List[RunMetricsRecord]: ...