# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import time

from intergrax.legacy.rag_answers.contracts.pipeline_trace import PipelineTrace

__all__ = ["PipelineTrace", "StepTimer"]


class StepTimer:

    def __init__(self) -> None:
        self._start = time.perf_counter()

    def stop_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000
