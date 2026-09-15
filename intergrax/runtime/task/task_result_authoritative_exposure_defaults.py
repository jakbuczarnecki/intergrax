# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Default authoritative exposure values for terminal TaskResult construction."""

from __future__ import annotations

from intergrax.contracts.decision_authoritative_exposure import (
    ExposureUnevaluated,
    ExposureUnevaluatedReason,
)


def terminal_task_result_exposure_no_decision_gate() -> ExposureUnevaluated:
    return ExposureUnevaluated(
        scope=None,
        reason=ExposureUnevaluatedReason.NO_DECISION_GATE,
    )
