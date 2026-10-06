# © Artur Czarnecki. All rights reserved.

"""TRACE-X-P4-R2 mechanical qualification gates (Q02, Q03, Q17–Q36 subset)."""

from __future__ import annotations

import ast
import inspect
import subprocess
from pathlib import Path

import pytest

from intergrax.context.contracts import AssembledContext
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.events.payloads.canonical import (
    ContextAssemblyPayloadV1,
    ContextAssemblyPayloadV2,
    ContextAssemblyPayloadV3,
    ContextAssemblyPayloadV4,
    LlmCallPayloadV1,
    LlmCallPayloadV2,
    LlmCallPayloadV3,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.llm import model_call_attribution as mca
from intergrax.runtime.llm import model_context_attribution as mctx
from tests.qualification.trace_x._trace_x_p4_support import (
    CONTEXT_ASSEMBLY_SURFACE_INVENTORY,
    P4_R2_GATE_REGISTRY,
    P4_R3_GATE_REGISTRY,
    TRACE_X_P4_R2_START_HEAD,
    attribute_model_call_to_context,
    discovered_context_assembly_production_paths,
    gate_nodeids,
    nodeid_observed,
    observed_gate_passed,
)
from tests.qualification.trace_x.test_trace_x_p4_model_context_attribution import governed_execution
from tests.qualification.trace_x.test_trace_x_p4_model_context_attribution import (
    _record_primary_model_call,
)
from tests.qualification.trace_x.test_trace_x_p4_model_context_attribution import (
    record_context_assembled_from_engine,
)

pytestmark = [pytest.mark.qualification, pytest.mark.gate]


def test_txp4r2_q01_start_head_ancestry() -> None:
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", TRACE_X_P4_R2_START_HEAD, "HEAD"],
    )


def test_txp4r2_q02_model_call_surfaces_classified() -> None:
    from tests.qualification.trace_x.test_trace_x_p4_r3_closed_world import (
        test_txp4r3_q02_model_call_surfaces_closed_world_classified,
    )

    test_txp4r3_q02_model_call_surfaces_closed_world_classified()


def test_txp4r2_q03_context_assembly_surfaces_classified() -> None:
    discovered = discovered_context_assembly_production_paths()
    classified = {row.path for row in CONTEXT_ASSEMBLY_SURFACE_INVENTORY}
    unknown = discovered - classified
    assert not unknown, f"unclassified context assembly surfaces: {sorted(unknown)}"


def test_txp4r2_q17_wrong_event_id_rejected(governed_execution) -> None:
    run_id, task_id, _a, _e, bus = governed_execution
    messages = (ChatMessage(role="user", content="id"),)
    assembled = AssembledContext(
        messages=messages,
        fragments_included=(),
        fragments_excluded=(),
        provenance=(),
        total_tokens=1,
        budget_tokens=100,
    )
    record_context_assembled_from_engine(bus, assembled=assembled, task_id=task_id, run_id=run_id)
    record_context_assembled_from_engine(bus, assembled=assembled, task_id=task_id, run_id=run_id)
    ctx_events = [e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED]
    ce1, ce2 = ctx_events[0], ctx_events[1]
    _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)
    llm = next(e for e in reversed(bus.history) if e.event_type == RuntimeEventType.LLM_CALL)
    verdict = attribute_model_call_to_context(ce1, llm)
    assert verdict.attributable is False
    assert verdict.reason == "context_event_id_mismatch"
    assert attribute_model_call_to_context(ce2, llm).attributable


def test_txp4r2_q18_wrong_input_hash_rejected(governed_execution) -> None:
    run_id, task_id, _a, _e, bus = governed_execution
    messages_a = (ChatMessage(role="user", content="a"),)
    messages_b = (ChatMessage(role="user", content="b"),)
    record_context_assembled_from_engine(
        bus,
        assembled=AssembledContext(
            messages=messages_a,
            fragments_included=(),
            fragments_excluded=(),
            provenance=(),
            total_tokens=1,
            budget_tokens=100,
        ),
        task_id=task_id,
        run_id=run_id,
    )
    ce = next(e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED)
    _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages_b)
    llm = next(e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL)
    verdict = attribute_model_call_to_context(ce, llm)
    assert verdict.attributable is False
    assert verdict.reason == "model_input_hash_mismatch"


def test_txp4r2_q19_fingerprint_contradiction_rejected(governed_execution) -> None:
    run_id, task_id, _a, _e, bus = governed_execution
    messages = (ChatMessage(role="user", content="fp"),)
    assembled = AssembledContext(
        messages=messages,
        fragments_included=(),
        fragments_excluded=(),
        provenance=(),
        total_tokens=1,
        budget_tokens=100,
    )
    record_context_assembled_from_engine(bus, assembled=assembled, task_id=task_id, run_id=run_id)
    ce = next(e for e in bus.history if e.event_type == RuntimeEventType.CONTEXT_ASSEMBLED)
    _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)
    llm = next(e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL)
    verdict = attribute_model_call_to_context(ce, llm, expected_context_fingerprint="not-the-real-fingerprint")
    assert verdict.attributable is False
    assert verdict.reason == "decision_fingerprint_contradiction"


def test_txp4r2_q21_provider_model_identity_on_llm_call(governed_execution) -> None:
    run_id, task_id, _a, _e, bus = governed_execution
    messages = (ChatMessage(role="user", content="pm"),)
    record_context_assembled_from_engine(
        bus,
        assembled=AssembledContext(
            messages=messages,
            fragments_included=(),
            fragments_excluded=(),
            provenance=(),
            total_tokens=1,
            budget_tokens=100,
        ),
        task_id=task_id,
        run_id=run_id,
    )
    _record_primary_model_call(bus, task_id=task_id, run_id=run_id, messages=messages)
    llm = next(e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL)
    from tests.qualification.trace_x._trace_x_p4_support import parse_llm_call_payload

    payload = parse_llm_call_payload(llm)
    assert payload is not None
    assert payload.provider.strip()
    assert payload.model.strip()


def test_txp4r2_q23_internal_scope_still_factual_not_primary_cert(governed_execution) -> None:
    from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

    run_id, task_id, _a, _e, bus = governed_execution
    messages = (ChatMessage(role="user", content="internal"),)
    _record_primary_model_call(
        bus,
        task_id=task_id,
        run_id=run_id,
        messages=messages,
        execution_scope=ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL,
    )
    llm = next(e for e in bus.history if e.event_type == RuntimeEventType.LLM_CALL)
    from tests.qualification.trace_x._trace_x_p4_support import parse_llm_call_payload

    payload = parse_llm_call_payload(llm)
    assert payload is not None
    assert payload.execution_scope == ModelCallExecutionScope.INTERNAL_OPTIMIZATION_CALL


def test_txp4r2_q24_generate_with_tools_uses_attribution_scope() -> None:
    from intergrax.runtime.llm.model_call_runtime_evidence_adapter import ModelCallRuntimeEvidenceAdapter

    src = inspect.getsource(ModelCallRuntimeEvidenceAdapter.generate_with_tools)
    assert "model_call_attribution_scope" in src


def test_txp4r2_q26_structured_output_uses_attribution_scope() -> None:
    from intergrax.runtime.llm.model_call_runtime_evidence_adapter import ModelCallRuntimeEvidenceAdapter

    src = inspect.getsource(ModelCallRuntimeEvidenceAdapter.generate_structured)
    assert "model_call_attribution_scope" in src


def test_txp4r2_q27_production_runtime_wraps_model_evidence_adapter() -> None:
    root = Path(__file__).resolve().parents[3]
    bridge = (root / "intergrax/applications/_shared/runtime_config_bridge.py").read_text(encoding="utf-8")
    assert "wrap_model_call_runtime_evidence" in bridge


def test_txp4r2_q28_production_runtime_binds_failure_evidence_recorder() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            "git",
            "grep",
            "-l",
            "bind_active_execution_evidence_context",
            "--",
            "intergrax/applications",
            "intergrax/runtime/execution",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert result.stdout.strip()


def test_txp4r2_q30_runtime_event_sole_factual_plane() -> None:
    root = Path(__file__).resolve().parents[3]
    for forbidden in (
        "ModelTraceStore",
        "ContextModelLinkStore",
        "PromptAuditStore",
    ):
        result = subprocess.run(
            ["git", "grep", forbidden, "--", "intergrax/runtime/llm"],
            cwd=root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1


def test_txp4r2_q31_context_engineering_fingerprint_owner() -> None:
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            "git",
            "grep",
            "compute_context_decision_evidence_fingerprint",
            "--",
            "intergrax/runtime/llm",
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if "model_context_attribution.py" not in line]
    assert not lines


def test_txp4r2_q32_no_heuristic_join_in_attribution() -> None:
    src = inspect.getsource(mctx.try_attribute_model_call_to_context)
    forbidden = (
        "nearest",
        "timestamp",
        "last_context",
        "same_run_only",
        "proximity",
    )
    lowered = src.lower()
    for term in forbidden:
        assert term not in lowered


def test_txp4r2_q33_payload_schema_compatibility() -> None:
    ContextAssemblyPayloadV1.model_validate(
        {"node_id": "n", "context_original_chars": 1, "context_final_chars": 1}
    )
    ContextAssemblyPayloadV2.model_validate(
        {"node_id": "n", "context_original_chars": 1, "context_final_chars": 1}
    )
    ContextAssemblyPayloadV3.model_validate(
        {
            "node_id": "n",
            "context_original_chars": 1,
            "context_final_chars": 1,
            "model_input_messages_hash": "h",
        }
    )
    ContextAssemblyPayloadV4.model_validate(
        {
            "node_id": "n",
            "context_original_chars": 1,
            "context_final_chars": 1,
            "model_input_messages_hash": "h",
            "context_decision_evidence_fingerprint": "f",
        }
    )
    LlmCallPayloadV1.model_validate({"model": "m"})
    LlmCallPayloadV2.model_validate(
        {
            "model": "m",
            "provider": "p",
            "model_input_messages_hash": "h",
            "execution_scope": "primary_model_call",
        }
    )
    LlmCallPayloadV3.model_validate(
        {
            "model": "m",
            "provider": "p",
            "model_input_messages_hash": "h",
            "execution_scope": "primary_model_call",
            "context_assembly_event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        }
    )


def test_txp4r2_q34_ce01_owner_regression() -> None:
    from tests.qualification.ce_01.test_ce_01_gates import test_ce_q1_canonical_context_engine_entry_surfaces

    test_ce_q1_canonical_context_engine_entry_surfaces()


def test_txp4r2_q35_p1_p2_p3_minimal_regression() -> None:
    from tests.qualification.trace_x._trace_x_p1_qualification_tests import test_txp1_q01_correct_start_head_provenance
    from tests.qualification.trace_x._trace_x_p2_qualification_tests import test_txp2_q01_correct_start_head_provenance
    from tests.qualification.trace_x._trace_x_p3_r1_qualification_tests import (
        test_txp3r1r1_q01_r1_start_head_ancestry,
    )

    test_txp1_q01_correct_start_head_provenance()
    test_txp2_q01_correct_start_head_provenance()
    test_txp3r1r1_q01_r1_start_head_ancestry()


def test_txp4r2_q36_frz_trc_05_readiness_registry_complete() -> None:
    import json
    import os

    gate_ids = {row.gate_id for row in P4_R3_GATE_REGISTRY}
    assert len(gate_ids) == len(P4_R3_GATE_REGISTRY)
    for row in P4_R3_GATE_REGISTRY:
        assert row.nodeids
    manifest = Path(".tmp/session/trace-x-p4-r3/pass1_observed_nodeids.json")
    if os.environ.get("TRACE_X_P4_PASS1") == "1":
        return
    if manifest.is_file():
        passed = set(json.loads(manifest.read_text(encoding="utf-8")))
        for row in P4_R3_GATE_REGISTRY:
            if not row.pass1_required:
                continue
            assert observed_gate_passed(row.gate_id, passed), row.gate_id


def test_txp4r2_q07_typed_execution_scope_in_attribution_scope() -> None:
    assert hasattr(mca, "get_model_call_execution_scope")


def test_txp4r2_registry_nodeids_exist_as_tests() -> None:
    root = Path(__file__).resolve().parent
    names: set[str] = set()
    for path in root.glob("test_trace_x_p4*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                names.add(f"{path.name}::{node.name}")
                names.add(node.name)
    missing = [nid for nid in gate_nodeids(P4_R3_GATE_REGISTRY) if not nodeid_observed(nid, names)]
    assert not missing, missing
