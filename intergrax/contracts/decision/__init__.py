# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Decision contracts namespace: MP-4B core (``decision.py``) and integration adapters."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from intergrax.contracts.decision.integration import (
    DecisionAdapterMetadata,
    DecisionIntegrationAdapterProvider,
    DecisionIntegrationAuditProvider,
    DecisionIntegrationAuditRecord,
    DecisionIntegrationResult,
    DecisionIntegrationStatus,
    DecisionLifecycleIntegrationAdapter,
    DecisionSystemIntegrationAdapter,
    DecisionSystemIntegrationEngine,
    PlatformDecisionLifecycleReference,
    REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE,
    ReferenceDecisionLifecycleReference,
    ReferenceEnterpriseLifecycleState,
)

_INTEGRATION_EXPORTS = (
    "DecisionAdapterMetadata",
    "DecisionIntegrationAdapterProvider",
    "DecisionIntegrationAuditProvider",
    "DecisionIntegrationAuditRecord",
    "DecisionIntegrationResult",
    "DecisionIntegrationStatus",
    "DecisionLifecycleIntegrationAdapter",
    "DecisionSystemIntegrationAdapter",
    "DecisionSystemIntegrationEngine",
    "PlatformDecisionLifecycleReference",
    "REFERENCE_DECISION_LIFECYCLE_SOURCE_TYPE",
    "ReferenceDecisionLifecycleReference",
    "ReferenceEnterpriseLifecycleState",
)

_MP4B_EXPORTS = (
    "SCHEMA_DECISION_V1",
    "SCHEMA_DECISION_OUTCOME_V1",
    "SCHEMA_DECISION_REFERENCES_V1",
    "DecisionId",
    "Decision",
    "DecisionContractInvariantError",
    "DecisionScopeInvariantError",
    "DecisionIdentityInvariantError",
    "DecisionLifecycleTransitionError",
    "DecisionProvenanceInvariantError",
    "DecisionLifecycleState",
    "DecisionOutcomeDisposition",
    "DecisionOutcome",
    "DecisionReferences",
    "DecisionLifecycleTransition",
    "mint_decision_id",
    "validate_decision_id",
    "validate_decision_scope",
    "validate_decision_identity",
    "validate_decision_provenance",
    "validate_decision_transition",
)


def _load_mp4b_decision_module() -> object:
    """Load sibling ``decision.py``; this package shadows that module path."""
    core_path = Path(__file__).resolve().parent.parent / "decision.py"
    module_name = "intergrax.contracts._decision_mp4b"
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load MP-4B decision contracts from {core_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_mp4b = _load_mp4b_decision_module()
for _name in _MP4B_EXPORTS:
    globals()[_name] = getattr(_mp4b, _name)

__all__ = [* _INTEGRATION_EXPORTS, *_MP4B_EXPORTS]
