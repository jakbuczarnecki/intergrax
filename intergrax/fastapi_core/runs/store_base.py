# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from abc import ABC, abstractmethod

from intergrax.fastapi_core.runs.models import RunResponse


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
