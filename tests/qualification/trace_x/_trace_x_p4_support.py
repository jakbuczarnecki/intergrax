# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4 / P4-R2 qualification support (model ↔ context attribution)."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.runtime.events.payload_registry import validate_payload_envelope
from intergrax.runtime.events.payloads.canonical import (
    ContextAssemblyPayloadV3,
    ContextAssemblyPayloadV4,
    LlmCallPayloadV2,
    LlmCallPayloadV3,
)
from intergrax.runtime.llm.model_context_attribution import (
    ModelContextAttributionVerdict,
    try_attribute_model_call_to_context,
)

TRACE_X_P4_START_HEAD: Final[str] = "7d782af85fa97807b882cc9063a5d57aef899e57"
TRACE_X_P4_R2_START_HEAD: Final[str] = "54936ecf758e68d6b79f2b05604fca7c1fc79849"

_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class P4GateEvidence:
    gate_id: str
    description: str
    nodeids: tuple[str, ...]
    pass1_required: bool = True


@dataclass(frozen=True, slots=True)
class ClassifiedSurface:
    path: str
    category: str
    evidence_nodeid: str


MODEL_CALL_SURFACE_INVENTORY: Final[tuple[ClassifiedSurface, ...]] = (
    ClassifiedSurface(
        "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
        "generate_messages",
        "test_txp4r2_q24_generate_with_tools_uses_attribution_scope",
    ),
    ClassifiedSurface(
        "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
        "generate_with_tools",
        "test_txp4r2_q24_generate_with_tools_uses_attribution_scope",
    ),
    ClassifiedSurface(
        "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
        "generate_structured",
        "test_txp4r2_q26_structured_output_uses_attribution_scope",
    ),
    ClassifiedSurface(
        "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
        "stream_messages",
        "test_txp4r1_q25_streaming_not_production_primary",
    ),
    ClassifiedSurface(
        "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
        "stream_with_tools",
        "test_txp4r1_q25_streaming_not_production_primary",
    ),
    ClassifiedSurface(
        "intergrax/runtime/token_optimization/llm_router.py",
        "internal optimization structured",
        "test_txp4_q16_internal_optimization_scope",
    ),
)

CONTEXT_ASSEMBLY_SURFACE_INVENTORY: Final[tuple[ClassifiedSurface, ...]] = (
    ClassifiedSurface(
        "intergrax/runtime/nexus/context/context_engine.py",
        "ContextEngine UAEP",
        "test_txp4_q12_primary_context_model_e2e_attribution",
    ),
    ClassifiedSurface(
        "intergrax/runtime/events/context_skill_recording.py",
        "record_context_assembled_from_engine",
        "test_txp4_q12_primary_context_model_e2e_attribution",
    ),
    ClassifiedSurface(
        "intergrax/runtime/events/context_skill_recording.py",
        "skill assembly CONTEXT_ASSEMBLED",
        "test_txp4_q12_primary_context_model_e2e_attribution",
    ),
    ClassifiedSurface(
        "intergrax/runtime/nexus/uaep/uaep_executor.py",
        "UAEP CONTEXT_ASSEMBLED",
        "test_txp4_q12_primary_context_model_e2e_attribution",
    ),
)


def discovered_model_call_production_paths() -> set[str]:
    rel_paths = (
        "intergrax/runtime/llm/model_call_runtime_evidence_adapter.py",
        "intergrax/runtime/token_optimization/llm_router.py",
    )
    discovered: set[str] = set()
    for rel in rel_paths:
        if (_REPO_ROOT / rel).is_file():
            discovered.add(rel)
    return discovered


def discovered_context_assembly_production_paths() -> set[str]:
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "record_context_assembled_from_engine",
            "--",
            "intergrax/runtime",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    paths = set()
    if result.returncode == 0:
        paths.update(line.strip() for line in result.stdout.splitlines() if line.strip())
    uaep = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "RuntimeEventType.CONTEXT_ASSEMBLED",
            "--",
            "intergrax/runtime/nexus/uaep/uaep_executor.py",
            "intergrax/runtime/events/context_skill_recording.py",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if uaep.returncode == 0:
        paths.update(line.strip() for line in uaep.stdout.splitlines() if line.strip())
    return paths


def gate_nodeids(registry: tuple[P4GateEvidence, ...]) -> tuple[str, ...]:
    out: list[str] = []
    for row in registry:
        out.extend(row.nodeids)
    return tuple(out)


def observed_gate_passed(gate_id: str, passed_nodeids: set[str]) -> bool:
    for row in P4_R2_GATE_REGISTRY:
        if row.gate_id != gate_id:
            continue
        return any(nodeid in passed_nodeids for nodeid in row.nodeids)
    return False


P4_R2_GATE_REGISTRY: tuple[P4GateEvidence, ...] = (
    P4GateEvidence("TXP4R2-Q01", "R2 START_HEAD ancestry", ("test_txp4r2_q01_start_head_ancestry",)),
    P4GateEvidence("TXP4R2-Q02", "model-call surfaces classified", ("test_txp4r2_q02_model_call_surfaces_classified",)),
    P4GateEvidence(
        "TXP4R2-Q03",
        "context assembly surfaces classified",
        ("test_txp4r2_q03_context_assembly_surfaces_classified",),
    ),
    P4GateEvidence("TXP4R2-Q04", "recorder binding resets tenant", ("test_txp4r1_q04_recorder_binding_resets_tenant",)),
    P4GateEvidence("TXP4R2-Q05", "nested tenant binding", ("test_txp4r1_q05_nested_tenant_binding",)),
    P4GateEvidence(
        "TXP4R2-Q06",
        "sequential tenant isolation",
        ("test_txp4r1_q06_sequential_tenant_isolation",),
    ),
    P4GateEvidence("TXP4R2-Q07", "typed ModelCallExecutionScope", ("test_txp4r2_q07_typed_execution_scope_in_attribution_scope",)),
    P4GateEvidence("TXP4R2-Q08", "invalid execution_scope rejected", ("test_txp4r1_q08_invalid_execution_scope_rejected",)),
    P4GateEvidence(
        "TXP4R2-Q09",
        "deterministic decision fingerprint",
        ("test_txp4r1_q11_same_messages_different_decision_fingerprint",),
    ),
    P4GateEvidence("TXP4R2-Q10", "no raw content in fingerprint", ("test_txp4r1_q10_fingerprint_excludes_raw_content",)),
    P4GateEvidence(
        "TXP4R2-Q11",
        "distinct decision chains",
        ("test_txp4r1_q11_same_messages_different_decision_fingerprint",),
    ),
    P4GateEvidence(
        "TXP4R2-Q12",
        "CONTEXT_ASSEMBLED stores fingerprint",
        ("test_txp4_q12_primary_context_model_e2e_attribution",),
    ),
    P4GateEvidence(
        "TXP4R2-Q13",
        "exact context EventId ref",
        ("test_txp4_q12_primary_context_model_e2e_attribution",),
    ),
    P4GateEvidence(
        "TXP4R2-Q14",
        "positive E2E attribution",
        ("test_txp4_q12_primary_context_model_e2e_attribution",),
    ),
    P4GateEvidence(
        "TXP4R2-Q15",
        "identical hash different assembly",
        ("test_txp4r1_q15_same_execution_two_assemblies_distinguishable",),
    ),
    P4GateEvidence(
        "TXP4R2-Q16",
        "same-run multi-execution isolation",
        ("test_txp4_q13_same_run_multi_execution_isolation",),
    ),
    P4GateEvidence("TXP4R2-Q17", "wrong EventId rejected", ("test_txp4r2_q17_wrong_event_id_rejected",)),
    P4GateEvidence("TXP4R2-Q18", "wrong hash rejected", ("test_txp4r2_q18_wrong_input_hash_rejected",)),
    P4GateEvidence(
        "TXP4R2-Q19",
        "fingerprint contradiction rejected",
        ("test_txp4r2_q19_fingerprint_contradiction_rejected",),
    ),
    P4GateEvidence("TXP4R2-Q20", "cross-tenant rejected", ("test_txp4r1_q20_cross_tenant_attribution_rejected",)),
    P4GateEvidence(
        "TXP4R2-Q21",
        "provider/model identity",
        ("test_txp4r2_q21_provider_model_identity_on_llm_call",),
    ),
    P4GateEvidence(
        "TXP4R2-Q22",
        "PRIMARY attribution",
        ("test_txp4_q12_primary_context_model_e2e_attribution",),
    ),
    P4GateEvidence(
        "TXP4R2-Q23",
        "INTERNAL optimization classification",
        ("test_txp4r2_q23_internal_scope_still_factual_not_primary_cert",),
    ),
    P4GateEvidence(
        "TXP4R2-Q24",
        "generate_with_tools parity",
        ("test_txp4r2_q24_generate_with_tools_uses_attribution_scope",),
    ),
    P4GateEvidence(
        "TXP4R2-Q25",
        "streaming not production primary",
        ("test_txp4r1_q25_streaming_not_production_primary",),
    ),
    P4GateEvidence(
        "TXP4R2-Q26",
        "structured output parity",
        ("test_txp4r2_q26_structured_output_uses_attribution_scope",),
    ),
    P4GateEvidence(
        "TXP4R2-Q27",
        "production recorder wiring",
        ("test_txp4r2_q27_production_runtime_wraps_model_evidence_adapter",),
    ),
    P4GateEvidence(
        "TXP4R2-Q28",
        "production evidence context wiring",
        ("test_txp4r2_q28_production_runtime_binds_failure_evidence_recorder",),
    ),
    P4GateEvidence("TXP4R2-Q29", "no raw prompt in evidence", ("test_txp4_q22_no_raw_prompt_in_evidence",)),
    P4GateEvidence(
        "TXP4R2-Q30",
        "RuntimeEvent sole factual plane",
        ("test_txp4r2_q30_runtime_event_sole_factual_plane",),
    ),
    P4GateEvidence(
        "TXP4R2-Q31",
        "Context Engineering fingerprint owner",
        ("test_txp4r2_q31_context_engineering_fingerprint_owner",),
    ),
    P4GateEvidence("TXP4R2-Q32", "no heuristic join", ("test_txp4r2_q32_no_heuristic_join_in_attribution",)),
    P4GateEvidence("TXP4R2-Q33", "payload compatibility", ("test_txp4r2_q33_payload_schema_compatibility",)),
    P4GateEvidence("TXP4R2-Q34", "CE-01 regression", ("test_txp4r2_q34_ce01_owner_regression",)),
    P4GateEvidence("TXP4R2-Q35", "P1/P2/P3 regression", ("test_txp4r2_q35_p1_p2_p3_minimal_regression",)),
    P4GateEvidence(
        "TXP4R2-Q36",
        "FRZ-TRC-05 readiness",
        ("test_txp4r2_q36_frz_trc_05_readiness_registry_complete",),
    ),
    P4GateEvidence(
        "TXP4R2-LIFE-01",
        "context ref cleared after success",
        ("test_txp4r2_life_01_successful_model_call_clears_context_ref",),
    ),
    P4GateEvidence(
        "TXP4R2-LIFE-02",
        "context ref cleared after provider failure",
        ("test_txp4r2_life_02_provider_failure_clears_context_ref",),
    ),
    P4GateEvidence(
        "TXP4R2-LIFE-03",
        "no recorder cannot leak context ref",
        ("test_txp4r2_life_03_no_recorder_path_cannot_leak_context_ref",),
    ),
    P4GateEvidence(
        "TXP4R2-LIFE-04",
        "missing evidence context cannot leak",
        ("test_txp4r2_life_04_missing_evidence_context_cannot_leak_context_ref",),
    ),
    P4GateEvidence(
        "TXP4R2-LIFE-05",
        "sequential execution stale-ref prevention",
        ("test_txp4r2_life_05_sequential_executions_cannot_inherit_stale_ce",),
    ),
    P4GateEvidence(
        "TXP4R2-LIFE-06",
        "nested scope restores outer relation",
        ("test_txp4r2_life_06_nested_model_attribution_scope_restores_outer_relation",),
    ),
)

P4_R1_GATE_REGISTRY = P4_R2_GATE_REGISTRY


def parse_context_assembly_payload(
    event: RuntimeEvent,
) -> ContextAssemblyPayloadV4 | ContextAssemblyPayloadV3 | None:
    typed = validate_payload_envelope(event.payload)
    if isinstance(typed, (ContextAssemblyPayloadV4, ContextAssemblyPayloadV3)):
        return typed
    return None


def parse_llm_call_payload(event: RuntimeEvent) -> LlmCallPayloadV3 | LlmCallPayloadV2 | None:
    typed = validate_payload_envelope(event.payload)
    if isinstance(typed, (LlmCallPayloadV3, LlmCallPayloadV2)):
        return typed
    return None


def attribute_model_call_to_context(
    context_event: RuntimeEvent,
    llm_event: RuntimeEvent,
    *,
    expected_context_fingerprint: str | None = None,
) -> ModelContextAttributionVerdict:
    return try_attribute_model_call_to_context(
        context_event,
        llm_event,
        expected_context_fingerprint=expected_context_fingerprint,
    )


__all__ = [
    "CONTEXT_ASSEMBLY_SURFACE_INVENTORY",
    "MODEL_CALL_SURFACE_INVENTORY",
    "P4GateEvidence",
    "P4_R1_GATE_REGISTRY",
    "P4_R2_GATE_REGISTRY",
    "TRACE_X_P4_R2_START_HEAD",
    "TRACE_X_P4_START_HEAD",
    "attribute_model_call_to_context",
    "discovered_context_assembly_production_paths",
    "discovered_model_call_production_paths",
    "gate_nodeids",
    "observed_gate_passed",
    "parse_context_assembly_payload",
    "parse_llm_call_payload",
]
