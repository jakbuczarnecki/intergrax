# © Artur Czarnecki. All rights reserved.

"""L0 deterministic critic gateway — Phase CRIT-V-3.2."""

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.critic.contracts import CriticLayer, CriticRequest, LayerVerdict
from intergrax.runtime.nexus.validation.validation_engine import NexusValidationEngine


class L0Gateway:
    """Wraps ``NexusValidationEngine`` for CVL L0 verification."""

    def __init__(self, *, engine: NexusValidationEngine | None = None) -> None:
        self._engine = engine or NexusValidationEngine()

    def verify(
        self,
        request: CriticRequest,
        *,
        contract: AgentContract,
        capability: str | None = None,
        plan_criteria: list[str] | None = None,
    ) -> LayerVerdict:
        execution = request.execution
        if execution is None:
            return LayerVerdict(
                layer=CriticLayer.L0_DETERMINISTIC,
                passed=False,
                errors=["missing execution payload for L0 verification"],
            )

        result = self._engine.validate(
            execution,
            contract=contract,
            capability=capability,
            plan_criteria=plan_criteria,
        )
        return LayerVerdict(
            layer=CriticLayer.L0_DETERMINISTIC,
            passed=result.valid,
            score=1.0 if result.valid else 0.0,
            errors=list(result.errors),
            warnings=list(result.warnings),
        )
