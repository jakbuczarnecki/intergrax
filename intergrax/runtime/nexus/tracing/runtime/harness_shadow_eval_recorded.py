# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from intergrax.runtime.nexus.tracing.trace_models import DiagnosticPayload


@dataclass(frozen=True)
class HarnessShadowEvalRecordedDiagV1(DiagnosticPayload):
    run_id: str = ""
    agent_id: str = ""
    scenario_id: str = ""
    passed: bool = False
    score: float = 0.0
    observation_id: str = ""

    def redact(self) -> HarnessShadowEvalRecordedDiagV1:
        return self

    @classmethod
    def schema_id(cls) -> str:
        return "intergrax.diag.runtime.harness_shadow_eval_recorded"

    @classmethod
    def schema_version(cls) -> int:
        return 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "scenario_id": self.scenario_id,
            "passed": self.passed,
            "score": self.score,
            "observation_id": self.observation_id,
        }
