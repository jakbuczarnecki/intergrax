# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.replay.replay_engine import ReplayEngine
from intergrax.runtime.replay.models import ReconstructedRun


class ReplayService:
    """
    Execution inspection capability.

    Provides high-level entry point for run reconstruction.
    """

    def __init__(self, engine: ReplayEngine) -> None:
        self._engine = engine

    def inspect_run(self, tenant_id: str, run_id: str) -> ReconstructedRun:
        return self._engine.reconstruct(tenant_id, run_id)
