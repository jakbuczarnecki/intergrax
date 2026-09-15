# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical composition for Decision exposure selection strategies (P0-B-D1-I1-A-R1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.decision_exposure_selection import DecisionExposureSelectionStrategy
from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.core.plugins.selection_ref import PlatformPluginSelectionRef
from intergrax.runtime.decision_plugin_composition import (
    DecisionExposureSelectionPluginLoadOutcome,
    load_decision_exposure_selection_strategy_plugin,
)
from intergrax.runtime.decision_plugin_policy import DecisionPluginLoadPolicy
from intergrax.runtime.execution.host_terminal_decision_exposure_selector import (
    HOST_TERMINAL_DECISION_EXPOSURE_SELECTOR_ID,
)


class DecisionExposureSelectionCompositionError(ValueError):
    """Raised when exposure selection plugin composition fails closed."""


@dataclass(frozen=True, slots=True)
class DecisionExposureSelectionComposition:
    """Immutable composed exposure selection strategy for one host."""

    strategy: DecisionExposureSelectionStrategy[object]
    report: DomainPluginLoadReport
    activated_plugin_id: str | None


def _composition_errors(report: DomainPluginLoadReport) -> tuple[str, ...]:
    errors: list[str] = []
    for item in report.failed:
        errors.append(f"exposure selection plugin load failed: {item.spec.name}: {item.error}")
    for item in report.rejected:
        if item.fail_closed:
            errors.append(
                "exposure selection plugin admission rejected: "
                f"{item.spec.name}: {item.reason_code.value}",
            )
    return tuple(errors)


def compose_decision_exposure_selection(
    *,
    policy: DecisionPluginLoadPolicy | None = None,
    selection_ref: PlatformPluginSelectionRef | None = None,
) -> DecisionExposureSelectionComposition:
    """Compose built-in default or one admitted external exposure selection strategy."""
    chosen = policy if policy is not None else DecisionPluginLoadPolicy()
    outcome = load_decision_exposure_selection_strategy_plugin(
        policy=chosen,
        selection_ref=selection_ref,
    )
    if outcome.strategy is None:
        errors = _composition_errors(outcome.report)
        detail = "; ".join(errors) if errors else "exposure selection plugin composition failed"
        raise DecisionExposureSelectionCompositionError(detail)
    activated = (
        None
        if outcome.strategy.strategy_id == HOST_TERMINAL_DECISION_EXPOSURE_SELECTOR_ID
        else outcome.strategy.strategy_id
    )
    if selection_ref is not None and activated != selection_ref.plugin_id:
        raise DecisionExposureSelectionCompositionError(
            "requested exposure selection plugin was not activated",
        )
    return DecisionExposureSelectionComposition(
        strategy=outcome.strategy,
        report=outcome.report,
        activated_plugin_id=activated,
    )
