# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from dataclasses import dataclass

from intergrax.fastapi_core.execution.decisions.decisions import ExecutionDecision
from intergrax.fastapi_core.execution.failures.failures import FailureCategory


@dataclass(frozen=True)
class ExecutionDecisionRecord:
    run_id: str
    category: FailureCategory
    decision: ExecutionDecision
    attempt: int
    message: str | None = None
