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


class Gr13MatrixResult(StrEnum):
    QUALIFIED = "QUALIFIED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


GR13_BASELINE_HEAD: Final[str] = "4e5878f538521ef2bc7e080599574e23d8821a5a"
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
    positive_proof_nodes: tuple[str, ...]
    authority_proof_nodes: tuple[str, ...]
    result: Gr13MatrixResult


def _nid(path: str, name: str) -> str:
    return f"{path}::{name}"


_GR13_UNIT = (
    "tests/unit/runtime/governance/test_gr13_governance_evidence_gep_emission.py"
)
_GR8_SPINE = "tests/unit/runtime/governance/test_gr8_governance_evidence_spine.py"
_GR8_INFERENCE = "tests/unit/runtime/execution/test_inference_executor.py"


def _agentic_rows() -> tuple[Gr13ProofMatrixRow, ...]:
    rows: list[Gr13ProofMatrixRow] = []
    for deferred in GR13_AGENTIC_GOVERNANCE_EVIDENCE_DEFERRED:
        sem = gr10_agentic_gep_semantics(deferred.gep)
        emission = {
            "PRE_MODEL": "intergrax/runtime/policy/pre_model_policy_evaluation.py",
            "AGENT_DECISION": "intergrax/runtime/interrupts/handler.py",
            "INTERRUPT": "intergrax/runtime/interrupts/handler.py",
            "TOOL_PLAN_OR_ACCESS": "intergrax/runtime/nexus/tools/tool_runtime.py",
            "TOOL_INVOCATION_POLICY": "intergrax/runtime/nexus/tools/invoker.py",
            "PRE_OUTPUT": "intergrax/runtime/kernel/step_kernel.py",
            "POST_RUN": "intergrax/runtime/governance/post_run_governance_bridge.py",
        }[deferred.gep]
        rows.append(
            Gr13ProofMatrixRow(
                "AGENTIC",
                deferred.gep,
                sem.canonical_owner,
                sem.production_path,
                deferred.deferred_proof,
                GR13_FACT_CONTRACT,
                GR13_PERSISTENCE_CONTRACT,
                emission,
                (
                    _nid(
                        _GR13_UNIT,
                        f"test_gr13_agentic_{deferred.gep.lower()}_emits_fact",
                    ),
                    *(
                        (
                            _nid(
                                _GR8_INFERENCE,
                                "test_inference_pre_model_allow_invokes_provider_once",
                            ),
                        )
                        if deferred.gep == "PRE_MODEL"
                        else ()
                    ),
                ),
                (_nid(_GR13_UNIT, "test_gr13_evidence_does_not_grant_permission"),),
                Gr13MatrixResult.QUALIFIED,
            ),
        )
    return tuple(rows)


def _orchestration_rows() -> tuple[Gr13ProofMatrixRow, ...]:
    rows: list[Gr13ProofMatrixRow] = []
    for deferred in GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED:
        sem = gr10_orchestration_gep_semantics(deferred.gep)
        emission = {
            "PRE_MODEL": "intergrax/runtime/nexus/orchestration/planning_runner.py",
            "TOOL_PLAN_OR_ACCESS": "intergrax/runtime/nexus/tools/tool_runtime.py",
            "TOOL_INVOCATION_POLICY": "intergrax/runtime/nexus/tools/invoker.py",
            "PRE_OUTPUT": "intergrax/runtime/policy/pre_output_policy_bridge.py",
            "POST_RUN": "intergrax/runtime/governance/post_run_governance_bridge.py",
        }[deferred.gep]
        rows.append(
            Gr13ProofMatrixRow(
                "ORCHESTRATION",
                deferred.gep,
                sem.canonical_owner,
                sem.production_path,
                deferred.deferred_proof,
                GR13_FACT_CONTRACT,
                GR13_PERSISTENCE_CONTRACT,
                emission,
                (
                    _nid(
                        _GR13_UNIT,
                        f"test_gr13_orchestration_{deferred.gep.lower()}_emits_fact",
                    ),
                ),
                (_nid(_GR13_UNIT, "test_gr13_evidence_does_not_grant_permission"),),
                Gr13MatrixResult.QUALIFIED,
            ),
        )
    return tuple(rows)


GR13_PROOF_MATRIX: Final[tuple[Gr13ProofMatrixRow, ...]] = (
    *_agentic_rows(),
    *_orchestration_rows(),
)


def gr13_all_proof_nodes() -> tuple[str, ...]:
    nodes: list[str] = []
    for row in GR13_PROOF_MATRIX:
        nodes.extend(row.positive_proof_nodes)
        nodes.extend(row.authority_proof_nodes)
    return tuple(dict.fromkeys(nodes))
