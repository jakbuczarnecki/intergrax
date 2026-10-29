# © Artur Czarnecki. All rights reserved.

"""ROADMAP-REPLAY-X closed-world constants @ AUDITED_HEAD."""

from __future__ import annotations

ROADMAP_REPLAY_X_AUDITED_HEAD = "a2ed11f0e811446ae67fdfd57b4c4997c546e10f"
DUP_X_ACCEPTANCE_SHA = "a7d83925b375b5f08eb0f175ced2c17fa026bba9"

FRZ_ROW_COUNT = 225
FRZ_PASS_COUNT = 95
FRZ_OPEN_COUNT = 130

HISTORICAL_LEDGER_ROW_COUNT = 232
HISTORICAL_UNIQUE_STAGE_MIN = 225

NPSC_42_MODULE_PATHS: tuple[str, ...] = (
    "intergrax/applications/_shared/harness_host_runtime.py",
    "intergrax/applications/_shared/scenario_runtime_baseline.py",
    "intergrax/runtime/execution/application_host_orchestration_composition.py",
    "intergrax/runtime/execution/environment_host_task_execution.py",
    "intergrax/runtime/execution/environment_orchestration_materialization.py",
    "intergrax/runtime/execution/harness_host_internal_composition.py",
    "intergrax/runtime/execution/harness_host_orchestration_wiring.py",
    "intergrax/runtime/execution/harness_host_topology_submission_wiring.py",
    "intergrax/runtime/execution/host_orchestration_assembly_validation.py",
    "intergrax/runtime/execution/worker_host_task_execution_composition.py",
)

MANDATORY_QUAL_DOCS: tuple[str, ...] = (
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_SHA_LADDER.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_HISTORICAL_STAGE_MATRIX.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_TASK_MATRIX.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_QUALIFICATION_MANIFEST.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_NPSC_4_2_REPLAY_DISPOSITION.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_C0_ROADMAP_DRIFT_AUDIT_RECORD.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CLOSURE_MATRIX.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROTOCOL_03_INDEPENDENT_AUDIT_PACKAGE.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ECAP_CURRENT_HEAD_AUDIT_REPORT.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_VBN_DELTA_RECONCILIATION.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PARENT_IMPLEMENTATION_DOSSIER.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_PROD_QUAL_DRIFT_LOG.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CLOSED_WORLD_INVENTORY.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_SECTION5_LEDGER_ROW_COUNT_RECONCILIATION.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_FRZ_PASS_IMPACT_MATRIX.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_VENDOR_BACKEND_DEPENDENCY_INVENTORY.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CORE_HARDCODING_FINDINGS_REGISTER.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CROSS_LAYER_REPLACEABILITY_MATRIX.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ALLOWED_FIXED_SUBSTRATE_DECISIONS.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_VENDOR_SELECTION_REACHABILITY_EVIDENCE.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_ALTERNATIVE_BACKEND_BEHAVIORAL_PROOF.md",
    "docs/project/maintainers/qualification/ROADMAP_REPLAY_X_CURRENT_HEAD_VENDOR_NEUTRALITY_AUDIT.md",
)
