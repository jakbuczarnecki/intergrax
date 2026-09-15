# © Artur Czarnecki. All rights reserved.

"""Test-only Decision exposure selection plugins (I1-A-R1 trust boundary proof)."""

from __future__ import annotations

from intergrax.contracts.decision_exposure_selection import (
    DecisionExposureCandidate,
    DecisionExposurePublicationPolicy,
    DecisionExposureSelectionDecision,
    DecisionExposureSelectionStrategy,
)
from intergrax.runtime.execution.host_terminal_decision_exposure_selector import (
    HostTerminalDecisionExposureSelector,
)

IMPORT_COUNTER: int = 0
IMPORT_COUNTER += 1


class ExposureSelectionTrustDelegate:
    """Valid external strategy delegating to the built-in host terminal selector."""

    @property
    def strategy_id(self) -> str:
        return "preload.exposure.trust.delegate"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> object:
        return HostTerminalDecisionExposureSelector().select(policy, candidates)


class ExposureSelectionTrustWrongRuntimeId:
    """Runtime strategy_id intentionally mismatched for identity tests."""

    @property
    def strategy_id(self) -> str:
        return "preload.exposure.trust.wrong_id"

    def select(
        self,
        policy: DecisionExposurePublicationPolicy,
        candidates: tuple[DecisionExposureCandidate[object], ...],
    ) -> object:
        return HostTerminalDecisionExposureSelector().select(policy, candidates)
