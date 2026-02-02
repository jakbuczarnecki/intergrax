# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from intergrax.fastapi_core.execution.models import ExecutionRequest
from intergrax.fastapi_core.execution.worker_contract import ExecutionWorker


class SimpleExecutionWorker(ExecutionWorker):
    """
    Minimal concrete worker used for tests and dry-run mode.

    Performs no business logic, just succeeds.
    """

    def execute(self, request: ExecutionRequest) -> None:
        return
