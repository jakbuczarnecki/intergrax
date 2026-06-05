# © Artur Czarnecki. All rights reserved.

"""Adaptive Harness Intelligence runtime package (Phase W-ADAPT-0.3).

Re-exports Phase V governance contracts from ``runtime.architecture.adaptive_governance``
so callers can import adaptive types from a single package without duplicating definitions.
"""

from __future__ import annotations

from intergrax.runtime.adaptive.contracts import (
    ADAPTIVE_PACKAGE_SCHEMA_VERSION,
    AdaptiveLifecycleMode,
    ProfileArtifactType,
    ProfileVersionStatus,
)
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveAuthorityLevel,
    AdaptiveGovernanceReport,
    AdaptiveLoopEnvelope,
    AdaptiveLoopGateResult,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
    build_default_adaptive_proposals,
    evaluate_adaptive_governance,
    evaluate_bounded_adaptive_loop,
)

__all__ = [
    "ADAPTIVE_PACKAGE_SCHEMA_VERSION",
    "AdaptiveAuthorityLevel",
    "AdaptiveGovernanceReport",
    "AdaptiveLifecycleMode",
    "AdaptiveLoopEnvelope",
    "AdaptiveLoopGateResult",
    "AdaptiveLoopKind",
    "AdaptiveLoopProposal",
    "ProfileArtifactType",
    "ProfileVersionStatus",
    "build_default_adaptive_proposals",
    "evaluate_adaptive_governance",
    "evaluate_bounded_adaptive_loop",
]
