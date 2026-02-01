# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from typing import Protocol

from fastapi import BackgroundTasks

from intergrax.fastapi_core.context import RequestContext
from intergrax.fastapi_core.runs.models import RunResponse


class RunService(Protocol):
    """
    Orchestration layer for run lifecycle.

    Responsible for:
    - creating runs
    - managing lifecycle
    - delegating persistence to RunStore
    """

    def create_run(self, context: RequestContext, background_tasks: BackgroundTasks) -> RunResponse:
        ...

    def get_run(self, run_id: str) -> RunResponse:
        ...

    def cancel_run(self, run_id: str) -> RunResponse:
        ...
