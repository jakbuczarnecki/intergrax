# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-0.5: Adaptive package import and contract smoke tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive import (
    ADAPTIVE_PACKAGE_SCHEMA_VERSION,
    AdaptiveLifecycleMode,
    AdaptiveLoopKind,
    AdaptiveLoopProposal,
    ProfileArtifactType,
    ProfileVersionStatus,
    build_default_adaptive_proposals,
    evaluate_bounded_adaptive_loop,
)
from intergrax.runtime.architecture import (
    AdaptiveLifecycleMode as ArchitectureAdaptiveLifecycleMode,
    ProfileVersionStatus as ArchitectureProfileVersionStatus,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_adaptive_package_schema_version() -> None:
    assert ADAPTIVE_PACKAGE_SCHEMA_VERSION == "1.0.0"


def test_adaptive_package_exports_verification_loop() -> None:
    from intergrax.runtime.adaptive import AdaptationEngine, AdaptationExecutor, AdaptationScheduler, VerificationLoop

    assert AdaptationEngine is not None
    assert AdaptationExecutor is not None
    assert AdaptationScheduler is not None
    assert VerificationLoop is not None


def test_lifecycle_modes_are_distinct() -> None:
    codes = {mode.value for mode in AdaptiveLifecycleMode}
    assert codes == {"l4_o", "l4_r", "l4_s", "l4_c", "l4_a", "l4_v"}


def test_profile_version_status_values() -> None:
    assert ProfileVersionStatus.ACTIVE.value == "active"
    assert ProfileArtifactType.RAG.value == "rag"


def test_governance_reexport_evaluates_default_proposal() -> None:
    proposals = build_default_adaptive_proposals()
    assert proposals
    result = evaluate_bounded_adaptive_loop(proposals[0])
    assert result.loop_id == proposals[0].envelope.loop_id


def test_architecture_module_reexports_adaptive_runtime_contracts() -> None:
    assert ArchitectureAdaptiveLifecycleMode.OBSERVE == AdaptiveLifecycleMode.OBSERVE
    assert ArchitectureProfileVersionStatus.DRAFT == ProfileVersionStatus.DRAFT


def test_adaptive_loop_kind_matches_governance_enum() -> None:
    assert AdaptiveLoopKind.ROUTING_TUNING.value == "routing_tuning"
    assert isinstance(build_default_adaptive_proposals()[0], AdaptiveLoopProposal)
