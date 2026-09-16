# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Evaluator-loop graph binding for runtime plan metadata (ORCH-2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from intergrax.runtime.nexus.execution.evaluator_loop_spec import EvaluatorLoopSpec


class EvaluatorLoopGraphBinding(BaseModel):
    """Standard evaluator-loop topology for product graph specs (AUDIT-IDEAL-10.1)."""

    model_config = ConfigDict(extra="forbid")

    producer_agent_id: str
    evaluator_agent_id: str
    revise_agent_id: str
    spec: EvaluatorLoopSpec
