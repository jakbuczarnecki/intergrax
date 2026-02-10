# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from dataclasses import dataclass

from intergrax.runtime.replay.metrics import ExecutionMetrics


@dataclass(slots=True)
class RunMetricsRecord:
    run_id: str
    agent_id: str
    metrics: ExecutionMetrics
