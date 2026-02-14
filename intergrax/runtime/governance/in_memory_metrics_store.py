# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.


from collections import defaultdict, deque
from typing import Dict, List

from intergrax.runtime.governance.contracts.metrics_record_dto import RunMetricsRecord
from intergrax.runtime.governance.contracts.metrics_store import ExecutionMetricsStore


class InMemoryMetricsStore(ExecutionMetricsStore):

    def __init__(self, capacity_per_agent: int = 1000) -> None:
        self._data: Dict[str, deque[RunMetricsRecord]] = defaultdict(
            lambda: deque(maxlen=capacity_per_agent)
        )

    def save(self, record: RunMetricsRecord) -> None:
        self._data[record.agent_id].append(record)

    def get_recent(self, agent_id: str, limit: int) -> List[RunMetricsRecord]:
        return list(self._data[agent_id])[-limit:]
