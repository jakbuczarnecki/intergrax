# © Artur Czarnecki. All rights reserved.

"""Built-in adaptive decision context providers (extend via new classes)."""

from __future__ import annotations

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    AdaptiveDataSourceKind,
    AdaptiveDataSourceRef,
    AdaptiveDecisionIntelligenceInput,
    HistoricalEvidence,
)
from testing_support.decision_e2e.model_matrix.enterprise_decision_lifecycle.contracts import (
    DecisionLifecycleState,
)


class LifecycleHistoryContextProvider:
    provider_id = "lifecycle_history"
    provider_version = "1"

    def contribute(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> tuple[HistoricalEvidence, ...]:
        if not input_data.lifecycle_records:
            return ()
        evidence: list[HistoricalEvidence] = []
        for record in input_data.lifecycle_records:
            refs = (
                AdaptiveDataSourceRef(
                    source_kind=AdaptiveDataSourceKind.LIFECYCLE_RECORD,
                    reference_id=record.decision_id,
                ),
            )
            summary = (
                f"Prior decision {record.decision_id} reached "
                f"{record.lifecycle_state.value}."
            )
            if record.lifecycle_state is DecisionLifecycleState.FAILED:
                summary = (
                    f"Prior decision {record.decision_id} failed — "
                    "manual correction may have been required."
                )
            evidence.append(
                HistoricalEvidence(
                    evidence_id=f"{self.provider_id}:{record.decision_id}",
                    summary=summary,
                    source_refs=refs,
                    context_provider_id=self.provider_id,
                    context_provider_version=self.provider_version,
                )
            )
        return tuple(evidence)


class AnalyticsHistoryContextProvider:
    provider_id = "analytics_history"
    provider_version = "1"

    def contribute(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> tuple[HistoricalEvidence, ...]:
        if not input_data.analytics_results:
            return ()
        evidence: list[HistoricalEvidence] = []
        for index, result in enumerate(input_data.analytics_results):
            ref_id = f"analytics:{index}:{result.payload_kind.value}"
            evidence.append(
                HistoricalEvidence(
                    evidence_id=f"{self.provider_id}:{ref_id}",
                    summary=(
                        f"Analytics ({result.payload_kind.value}) available for "
                        f"decisions: {', '.join(result.audit.decision_ids) or 'n/a'}."
                    ),
                    source_refs=(
                        AdaptiveDataSourceRef(
                            source_kind=AdaptiveDataSourceKind.ANALYTICS_RESULT,
                            reference_id=ref_id,
                        ),
                    ),
                    context_provider_id=self.provider_id,
                    context_provider_version=self.provider_version,
                )
            )
        return tuple(evidence)


class OptimizationContextProvider:
    provider_id = "optimization_context"
    provider_version = "1"

    def contribute(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> tuple[HistoricalEvidence, ...]:
        opt = input_data.optimization_result
        if opt is None or not opt.suggestions:
            return ()
        summaries = "; ".join(item.rationale[:80] for item in opt.suggestions[:3])
        return (
            HistoricalEvidence(
                evidence_id=f"{self.provider_id}:{opt.optimization_task_id}",
                summary=f"Optimization loop produced suggestions: {summaries}",
                source_refs=(
                    AdaptiveDataSourceRef(
                        source_kind=AdaptiveDataSourceKind.OPTIMIZATION_RESULT,
                        reference_id=opt.optimization_task_id,
                    ),
                ),
                context_provider_id=self.provider_id,
                context_provider_version=self.provider_version,
            ),
        )


class CapabilityContextProvider:
    provider_id = "capability_context"
    provider_version = "1"

    def contribute(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> tuple[HistoricalEvidence, ...]:
        if not input_data.capability_profiles:
            return ()
        evidence: list[HistoricalEvidence] = []
        for profile in input_data.capability_profiles:
            model_id = profile.model_identity.profile_key
            evidence.append(
                HistoricalEvidence(
                    evidence_id=f"{self.provider_id}:{model_id}",
                    summary=(
                        f"Capability profile for model {model_id} "
                        f"({len(profile.limitations)} limitation observations)."
                    ),
                    source_refs=(
                        AdaptiveDataSourceRef(
                            source_kind=AdaptiveDataSourceKind.CAPABILITY_PROFILE,
                            reference_id=model_id,
                        ),
                    ),
                    context_provider_id=self.provider_id,
                    context_provider_version=self.provider_version,
                )
            )
        return tuple(evidence)


class GovernanceContextProvider:
    provider_id = "governance_context"
    provider_version = "1"

    def contribute(
        self,
        input_data: AdaptiveDecisionIntelligenceInput,
    ) -> tuple[HistoricalEvidence, ...]:
        if not input_data.governance_decisions:
            return ()
        evidence: list[HistoricalEvidence] = []
        for index, decision in enumerate(input_data.governance_decisions):
            subject = decision.audit_metadata.recommended_profile_key or "n/a"
            evidence.append(
                HistoricalEvidence(
                    evidence_id=f"{self.provider_id}:{index}",
                    summary=(
                        f"Governance disposition {decision.disposition.value} "
                        f"for subject {subject}."
                    ),
                    source_refs=(
                        AdaptiveDataSourceRef(
                            source_kind=AdaptiveDataSourceKind.GOVERNANCE_DECISION,
                            reference_id=f"governance:{index}",
                        ),
                    ),
                    context_provider_id=self.provider_id,
                    context_provider_version=self.provider_version,
                )
            )
        return tuple(evidence)


def default_context_providers() -> tuple[
    LifecycleHistoryContextProvider,
    AnalyticsHistoryContextProvider,
    OptimizationContextProvider,
    CapabilityContextProvider,
    GovernanceContextProvider,
]:
    return (
        LifecycleHistoryContextProvider(),
        AnalyticsHistoryContextProvider(),
        OptimizationContextProvider(),
        CapabilityContextProvider(),
        GovernanceContextProvider(),
    )


__all__ = [
    "AnalyticsHistoryContextProvider",
    "CapabilityContextProvider",
    "GovernanceContextProvider",
    "LifecycleHistoryContextProvider",
    "OptimizationContextProvider",
    "default_context_providers",
]
