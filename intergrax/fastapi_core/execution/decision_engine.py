# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.fastapi_core.execution.failures import FailureCategory
from intergrax.fastapi_core.execution.policies import ExecutionPolicy
from intergrax.fastapi_core.execution.decisions import ExecutionDecision


class ExecutionDecisionEngine:

    def decide(
        self,
        category: FailureCategory,
        attempt: int,
        policy: ExecutionPolicy,
    ) -> ExecutionDecision:

        if category == FailureCategory.CANCELED:
            return ExecutionDecision.IGNORE

        if category == FailureCategory.RETRYABLE and attempt < policy.max_retries:
            return ExecutionDecision.RETRY

        return ExecutionDecision.FAIL
