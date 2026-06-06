# © Artur Czarnecki. All rights reserved.

"""Wrap sub-engine candidates into governed proposal packages (Phase W-ADAPT-2.6)."""

from __future__ import annotations

from intergrax.runtime.adaptive.adaptation_models import (
    AdaptationProposalCandidate,
    AdaptationProposalPackage,
)
from intergrax.runtime.adaptive.governance_pipeline import AdaptationGovernancePipeline
from intergrax.runtime.adaptive.adaptation_models import AdaptationEngineContext


class ProposalBuilder:
    """Build and gate proposal packages from sub-engine candidates."""

    def __init__(self, governance_pipeline: AdaptationGovernancePipeline) -> None:
        self._governance_pipeline = governance_pipeline

    def build_package(
        self,
        candidate: AdaptationProposalCandidate,
        *,
        context: AdaptationEngineContext,
    ) -> AdaptationProposalPackage:
        return self._governance_pipeline.evaluate(candidate, context=context)
