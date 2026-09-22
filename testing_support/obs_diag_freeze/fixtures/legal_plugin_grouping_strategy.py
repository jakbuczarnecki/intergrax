# © Artur Czarnecki. All rights reserved.

"""Freeze-guard fixture: legal plugin implementing an existing grouping contract."""

from __future__ import annotations

from intergrax.runtime.diagnostics.problem_grouping import (
    ProblemGroupingMethod,
    ProblemGroupingStrategyCharacteristics,
    ProblemGroupingStrategyId,
    ProblemGroupingStrategyResult,
    ProblemGroupingStrategyVersion,
)


class ObsDiagFreezeLegalGroupingStrategy:
    """Test-only strategy — must not trip authority constructor freeze scans."""

    @property
    def strategy_id(self) -> ProblemGroupingStrategyId:
        return ProblemGroupingStrategyId("obs_diag_freeze_legal_fixture")

    @property
    def strategy_version(self) -> ProblemGroupingStrategyVersion:
        return ProblemGroupingStrategyVersion("1")

    @property
    def characteristics(self) -> ProblemGroupingStrategyCharacteristics:
        return ProblemGroupingStrategyCharacteristics(
            method=ProblemGroupingMethod.DETERMINISTIC,
            deterministic=True,
        )

    def group(
        self,
        inputs: tuple[object, ...],
    ) -> ProblemGroupingStrategyResult:
        _ = inputs
        raise NotImplementedError("fixture-only")
