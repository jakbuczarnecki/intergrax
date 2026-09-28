# © Artur Czarnecki. All rights reserved.

"""GR-13 proof matrix rows derived from GR-10 deferred inventories (not a second applicability SSOT)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tests.qualification.governance.strategy.catalog import (
    GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED,
    GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED,
    gr10_agentic_gep_semantics,
    gr10_orchestration_gep_semantics,
)


class Gr13ProofPathKind(StrEnum):
    SHARED_CANONICAL_PATH = "SHARED_CANONICAL_PATH"
    STRATEGY_SPECIFIC_PATH = "STRATEGY_SPECIFIC_PATH"


GR13_BASELINE_HEAD: Final[str] = "4e5878f538521ef2bc7e080599574e23d8821a5a"
GR11_ACCEPTED_HEAD: Final[str] = GR13_BASELINE_HEAD
GR12_ACCEPTED_HEAD: Final[str] = "03dde6c68a37ac0a8fe19cc5bcf683da8a3afc06"
GR13_ADR: Final[str] = (
    "docs/project/technical/adr/entries/2026-09-21/"
    "ADR-GR-10-003-gr10-gr13-governance-evidence-certification-scope.md"
)
GR13_FACT_CONTRACT: Final[str] = (
    "intergrax.contracts.governed_execution_governance_evidence.GovernanceDecisionEvidenceFact"
)
GR13_PERSISTENCE_CONTRACT: Final[str] = (
    "intergrax.contracts.governed_execution_governance_evidence.GovernanceEvidencePersistencePort"
)
GR13_RECORDER: Final[str] = (
    "intergrax.runtime.governance.governance_evidence_recorder.GovernanceEvidenceRecorder"
)

GR13_SCENARIO_Y_PROOF_NODES: Final[tuple[str, ...]] = (
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a4_unknown_host_state_separation.py::"
    "test_crash_ambiguity_intent_without_outcome_not_explicit_unknown",
    "applications/governed_contractor_application/tests/host/"
    "test_gr7_a7_r1_recovery_integrity.py::test_crash_ambiguity_forged_repeat_blocked",
)

GR13_GR8_EVIDENCE_FAILURE_PROOF_NODES: Final[tuple[str, ...]] = (
    "tests/unit/runtime/governance/test_gr8_governance_evidence_spine.py::"
    "test_persistence_failure_does_not_flip_deny_to_allow",
    "tests/unit/runtime/governance/test_gr8_governance_evidence_spine.py::"
    "test_persistence_failure_does_not_flip_require_human_to_allow",
    "tests/unit/runtime/governance/test_gr8_governance_evidence_spine.py::"
    "test_idempotent_replay_does_not_duplicate_facts",
)

GR13_HELPER_ONLY_FORBIDDEN_MARKERS: Final[tuple[str, ...]] = (
    "record_governance_policy_decision_evidence(",
    "record_governance_policy_decision_evidence_for_active_identity(",
    "record_tool_plan_or_access_evidence(",
)


@dataclass(frozen=True, slots=True)
class Gr13ProofMatrixRow:
    strategy: str
    gep: str
    canonical_owner: str
    production_path: str
    obligation: str
    fact_contract: str
    persistence_contract: str
    emission_module: str
    proof_path_kind: Gr13ProofPathKind
    canonical_path_markers: tuple[str, ...]
    positive_proof_nodes: tuple[str, ...]
    authority_proof_nodes: tuple[str, ...]
    negative_proof_nodes: tuple[str, ...]


def _nid(path: str, name: str) -> str:
    return f"{path}::{name}"


_GR13_UNIT = (
    "tests/unit/runtime/governance/test_gr13_governance_evidence_gep_emission.py"
)
_GR8_SPINE = "tests/unit/runtime/governance/test_gr8_governance_evidence_spine.py"
_GR8_INFERENCE = "tests/unit/runtime/execution/test_inference_executor.py"

_AGENTIC_EMISSION: Final[dict[str, str]] = {
    "PRE_MODEL": "intergrax/runtime/policy/pre_model_policy_evaluation.py",
    "AGENT_DECISION": "intergrax/runtime/interrupts/handler.py",
    "INTERRUPT": "intergrax/runtime/interrupts/handler.py",
    "TOOL_PLAN_OR_ACCESS": "intergrax/runtime/nexus/tools/tool_runtime.py",
    "TOOL_INVOCATION_POLICY": "intergrax/runtime/nexus/tools/invoker.py",
    "PRE_OUTPUT": "intergrax/runtime/kernel/step_kernel.py",
    "POST_RUN": "intergrax/runtime/governance/post_run_governance_bridge.py",
}

_ORCHESTRATION_EMISSION: Final[dict[str, str]] = {
    "PRE_MODEL": "intergrax/runtime/nexus/orchestration/planning_runner.py",
    "TOOL_PLAN_OR_ACCESS": "intergrax/runtime/nexus/tools/tool_runtime.py",
    "TOOL_INVOCATION_POLICY": "intergrax/runtime/nexus/tools/invoker.py",
    "PRE_OUTPUT": "intergrax/runtime/policy/pre_output_policy_bridge.py",
    "POST_RUN": "intergrax/runtime/governance/post_run_governance_bridge.py",
}

_AGENTIC_PATH_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "PRE_MODEL": ("enforce_pre_model_before_structured_inference",),
    "AGENT_DECISION": ("ExecutionInterruptHandler", "resolve_decision"),
    "INTERRUPT": ("ExecutionInterruptHandler", "resolve_decision"),
    "TOOL_PLAN_OR_ACCESS": ("ToolRuntime.invoke",),
    "TOOL_INVOCATION_POLICY": ("RuntimeToolInvoker", "invoke"),
    "PRE_OUTPUT": ("HarnessKernel.execute_step",),
    "POST_RUN": ("invoke_post_run_governance",),
}

_ORCHESTRATION_PATH_MARKERS: Final[dict[str, tuple[str, ...]]] = {
    "PRE_MODEL": ("NexusPlanningRunner", "run"),
    "TOOL_PLAN_OR_ACCESS": ("ToolRuntime.invoke",),
    "TOOL_INVOCATION_POLICY": ("RuntimeToolInvoker", "invoke"),
    "PRE_OUTPUT": ("apply_pre_output_policy",),
    "POST_RUN": ("invoke_post_run_governance",),
}

_SHARED_GEPS: Final[frozenset[str]] = frozenset(
    {"TOOL_PLAN_OR_ACCESS", "TOOL_INVOCATION_POLICY", "POST_RUN"},
)


def _authority_nodes() -> tuple[str, ...]:
    return (
        _nid(_GR13_UNIT, "test_gr13_evidence_does_not_grant_permission"),
        _nid(_GR13_UNIT, "test_gr13_require_human_still_blocks_after_evidence"),
        _nid(_GR13_UNIT, "test_gr13_recorder_absence_does_not_grant_permission"),
        _nid(
            _GR13_UNIT,
            "test_gr13_evidence_persistence_failure_does_not_widen_authority",
        ),
    )


def _negative_nodes_for_gep(gep: str) -> tuple[str, ...]:
    spine = (
        _nid(_GR13_UNIT, "test_gr13_evidence_does_not_grant_permission"),
        _nid(
            _GR13_UNIT,
            "test_gr13_evidence_persistence_failure_does_not_widen_authority",
        ),
    )
    if gep in {
        "TOOL_PLAN_OR_ACCESS",
        "TOOL_INVOCATION_POLICY",
        "PRE_MODEL",
        "PRE_OUTPUT",
    }:
        return spine
    if gep in {"AGENT_DECISION", "INTERRUPT"}:
        return (
            *spine,
            _nid(_GR13_UNIT, "test_gr13_require_human_still_blocks_after_evidence"),
        )
    return spine


def _agentic_rows() -> tuple[Gr13ProofMatrixRow, ...]:
    rows: list[Gr13ProofMatrixRow] = []
    for deferred in GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED:
        sem = gr10_agentic_gep_semantics(deferred.gep)
        gep = deferred.gep
        proof_kind = (
            Gr13ProofPathKind.SHARED_CANONICAL_PATH
            if gep in _SHARED_GEPS
            else Gr13ProofPathKind.STRATEGY_SPECIFIC_PATH
        )
        positive = (
            _nid(_GR13_UNIT, f"test_gr13_agentic_{gep.lower()}_emits_fact"),
            *(
                (
                    _nid(
                        _GR8_INFERENCE,
                        "test_inference_pre_model_allow_invokes_provider_once",
                    ),
                )
                if gep == "PRE_MODEL"
                else ()
            ),
        )
        rows.append(
            Gr13ProofMatrixRow(
                "AGENTIC",
                gep,
                sem.canonical_owner,
                sem.production_path,
                deferred.deferred_proof,
                GR13_FACT_CONTRACT,
                GR13_PERSISTENCE_CONTRACT,
                _AGENTIC_EMISSION[gep],
                proof_kind,
                _AGENTIC_PATH_MARKERS[gep],
                positive,
                _authority_nodes(),
                _negative_nodes_for_gep(gep),
            ),
        )
    return tuple(rows)


def _orchestration_rows() -> tuple[Gr13ProofMatrixRow, ...]:
    rows: list[Gr13ProofMatrixRow] = []
    for deferred in GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED:
        sem = gr10_orchestration_gep_semantics(deferred.gep)
        gep = deferred.gep
        proof_kind = (
            Gr13ProofPathKind.SHARED_CANONICAL_PATH
            if gep in _SHARED_GEPS
            else Gr13ProofPathKind.STRATEGY_SPECIFIC_PATH
        )
        rows.append(
            Gr13ProofMatrixRow(
                "ORCHESTRATION",
                gep,
                sem.canonical_owner,
                sem.production_path,
                deferred.deferred_proof,
                GR13_FACT_CONTRACT,
                GR13_PERSISTENCE_CONTRACT,
                _ORCHESTRATION_EMISSION[gep],
                proof_kind,
                _ORCHESTRATION_PATH_MARKERS[gep],
                (
                    _nid(
                        _GR13_UNIT,
                        f"test_gr13_orchestration_{gep.lower()}_emits_fact",
                    ),
                ),
                _authority_nodes(),
                _negative_nodes_for_gep(gep),
            ),
        )
    return tuple(rows)


GR13_PROOF_MATRIX: Final[tuple[Gr13ProofMatrixRow, ...]] = (
    *_agentic_rows(),
    *_orchestration_rows(),
)

GR13_REGISTERED_EMISSION_MODULES: Final[frozenset[str]] = frozenset(
    row.emission_module for row in GR13_PROOF_MATRIX
) | frozenset(
    {
        "intergrax/runtime/governance/governance_policy_decision_evidence_recording.py",
        "intergrax/runtime/governance/governance_evidence_recorder.py",
        "intergrax/runtime/governance/governance_evidence_composition.py",
    },
)


def gr13_all_proof_nodes() -> tuple[str, ...]:
    nodes: list[str] = []
    for row in GR13_PROOF_MATRIX:
        nodes.extend(row.positive_proof_nodes)
        nodes.extend(row.authority_proof_nodes)
        nodes.extend(row.negative_proof_nodes)
    nodes.extend(GR13_SCENARIO_Y_PROOF_NODES)
    nodes.extend(GR13_GR8_EVIDENCE_FAILURE_PROOF_NODES)
    return tuple(dict.fromkeys(nodes))


def gr13_positive_proof_test_name(strategy: str, gep: str) -> str:
    return f"test_gr13_{strategy.lower()}_{gep.lower()}_emits_fact"
