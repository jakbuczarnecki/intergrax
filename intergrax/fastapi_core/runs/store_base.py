# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
from intergrax.fastapi_core.runs.models import RunResponse, RunStatus


class RunStore(ABC):
    """
    Abstract run storage contract.

    API layer MUST depend only on this interface.
    """

    @abstractmethod
    def create(self) -> RunResponse:
        raise NotImplementedError

    @abstractmethod
    def get(self, run_id: str) -> RunResponse:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, run_id: str) -> RunResponse:
        raise NotImplementedError
    
    @abstractmethod
    def update_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        duration_ms: Optional[int] = None,
        result_payload: Optional[dict] = None,
    ) -> RunResponse:
        raise NotImplementedError
