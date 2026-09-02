# © Artur Czarnecki. All rights reserved.

"""Minimal side-effect probe behind the public authorization execute callback."""

from __future__ import annotations

from dataclasses import dataclass, field

from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest


@dataclass
class SideEffectExecutionRecord:
    side_effect_scope_id: str
    task_id: str
    run_id: str
    resource: str | None


@dataclass
class SideEffectProbe:
    """Records whether execution happened and proposal identity — no vendor logic."""

    executions: list[SideEffectExecutionRecord] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.executions)

    def execute(self, side_effect: MeaningfulSideEffectRequest) -> str:
        self.executions.append(
            SideEffectExecutionRecord(
                side_effect_scope_id=side_effect.side_effect_scope_id,
                task_id=side_effect.task_id,
                run_id=side_effect.run_id,
                resource=side_effect.resource,
            )
        )
        return "executed"
