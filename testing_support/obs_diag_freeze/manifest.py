# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG enterprise freeze — immutable qualification manifest (test-support SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

CERTIFIED_CODE_SHA = "c5acfc17ecea4d0e4b03405eabf735799c020ff8"

CERTIFICATION_RECORD_COMMIT = "150cf9a51833a1cf2ca89ded7e9447f3f9e2eea7"

CERTIFICATION_METADATA_RECONCILIATION_COMMIT = "f34c55accf23fa46633ded1803150de94f37770a"

STALE_CERTIFICATION_RECORD_COMMIT = "594bec4c2215f64b73f5758531c89c537a12078d"


class ObsDiagFrozenInvariantId(StrEnum):
    OBSERVABILITY_OWNS_EVIDENCE_FACTS = "observability_owns_evidence_facts"
    DIAGNOSTICS_OWNS_INTERPRETATION = "diagnostics_owns_interpretation"
    EXECUTION_OWNS_EXECUTION_IDENTITY = "execution_owns_execution_identity"
    FACTUAL_RECONSTRUCTION_IS_SHARED = "factual_reconstruction_is_shared"
    ONE_DIAGNOSTIC_ORCHESTRATOR_AUTHORITY = "one_diagnostic_orchestrator_authority"
    ONE_PROBLEM_LIFECYCLE_ENGINE_AUTHORITY = "one_problem_lifecycle_engine_authority"
    ONE_DEFAULT_EXECUTION_RECONSTRUCTOR_AUTHORITY = "one_default_execution_reconstructor_authority"
    NO_DUPLICATE_RUNTIME_EVENT_TRUTH = "no_duplicate_runtime_event_truth"
    NO_SECOND_EXECUTION_TREE = "no_second_execution_tree"
    NO_TIMESTAMP_EXECUTION_AUTHORITY = "no_timestamp_execution_authority"
    NO_VENDOR_TELEMETRY_AS_CANONICAL_TRUTH = "no_vendor_telemetry_as_canonical_truth"
    NO_BYPASS_CANONICAL_DIAGNOSTIC_COMPOSITION = "no_bypass_canonical_diagnostic_composition"


class ObsDiagChangeClass(StrEnum):
    SAFE_EXTENSION = "safe_extension"
    REQUALIFICATION_REQUIRED = "requalification_required"


class ObsDiagRequalificationSignal(StrEnum):
    CANONICAL_CONTRACT_CHANGE = "canonical_contract_change"
    OWNERSHIP_CHANGE = "ownership_change"
    NEW_AUTHORITY = "new_authority"
    NEW_BYPASS = "new_bypass"
    IDENTITY_SEMANTICS_CHANGE = "identity_semantics_change"
    PAYLOAD_SEMANTICS_CHANGE = "payload_semantics_change"
    RECONSTRUCTION_SEMANTICS_CHANGE = "reconstruction_semantics_change"


@dataclass(frozen=True, slots=True)
class ObsDiagAuthorityOwner:
    symbol: str
    defining_module: str
    direct_construction_allowed_modules: tuple[str, ...]


CANONICAL_AUTHORITY_OWNERS: tuple[ObsDiagAuthorityOwner, ...] = (
    ObsDiagAuthorityOwner(
        symbol="DiagnosticOrchestrator",
        defining_module="intergrax/runtime/diagnostics/diagnostic_orchestrator.py",
        direct_construction_allowed_modules=(
            "intergrax/applications/_shared/diagnostic_composition.py",
        ),
    ),
    ObsDiagAuthorityOwner(
        symbol="ProblemLifecycleEngine",
        defining_module="intergrax/runtime/diagnostics/problem_lifecycle.py",
        direct_construction_allowed_modules=(
            "intergrax/applications/_shared/diagnostic_composition.py",
        ),
    ),
    ObsDiagAuthorityOwner(
        symbol="ExecutionReconstructor",
        defining_module=(
            "intergrax/runtime/observability/reconstruction/execution_reconstruction.py"
        ),
        direct_construction_allowed_modules=(
            "intergrax/applications/_shared/diagnostic_composition.py",
            "intergrax/runtime/observability/historical_reconstruction.py",
        ),
    ),
)


FROZEN_CONTRACT_MODULES: tuple[str, ...] = (
    "intergrax/contracts/execution_evidence/persistence_port.py",
    "intergrax/contracts/functional_evidence/persistence.py",
    "intergrax/contracts/platform_causal_evidence.py",
    "intergrax/contracts/execution_reconstruction.py",
    "intergrax/contracts/diagnostics/problem_persistence.py",
    "intergrax/runtime/diagnostics/problem_occurrence_persistence.py",
    "intergrax/runtime/diagnostics/problem_grouping.py",
    "intergrax/runtime/integrations/observability.py",
    "intergrax/runtime/diagnostics/diagnostic_read_service.py",
    "intergrax/applications/_shared/diagnostic_composition.py",
)


FROZEN_COMPOSITION_WIRING_MODULES: tuple[str, ...] = (
    "intergrax/applications/_shared/diagnostic_composition.py",
    "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
)


OBS_DIAG_FROZEN_REUSED_GATE_MODULES: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_one_spine_diagnostic_orchestrator_gate.py",
    "tests/unit/runtime/architecture/test_one_spine_problem_store_gate.py",
    "tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py",
    "tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",
    "tests/unit/runtime/architecture/test_diag_foundation_4_entrypoint_consistency.py",
    "tests/unit/applications/_shared/test_obs_diag_x2a_canonical_host_diagnostic_composition.py",
    "tests/unit/applications/_shared/test_obs_diag_x2b_canonical_override_one_resolution.py",
    "tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py",
    "tests/unit/runtime/architecture/test_obs_diag_x5a_provider_evidence_integrity.py",
    "tests/unit/runtime/architecture/test_ue_10r2_single_canonical_root_execution_id_gate.py",
)


OBS_DIAG_A1_ARCHITECTURE_REGRESSION_TARGETS: tuple[str, ...] = (
    "tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py",
    "tests/unit/runtime/architecture/test_obs_diag_conformance_qualification.py",
    "tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py",
    "tests/unit/runtime/architecture/test_obs_diag_port_1_gates.py",
    "tests/unit/runtime/architecture/test_one_spine_diagnostic_orchestrator_gate.py",
    "tests/unit/runtime/architecture/test_one_spine_problem_store_gate.py",
    "tests/unit/applications/_shared/test_obs_diag_x2a_canonical_host_diagnostic_composition.py",
    "tests/unit/applications/_shared/test_obs_diag_x2b_canonical_override_one_resolution.py",
    "tests/unit/applications/_shared/test_obs_diag_x3_universal_spine_adoption.py",
)
