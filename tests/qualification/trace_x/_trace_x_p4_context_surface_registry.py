# © Artur Czarnecki. All rights reserved.

"""Explicit TRACE-X-P4-R4 context assembly / evidence surface registry."""

from __future__ import annotations

from typing import Final

from tests.qualification.trace_x._trace_x_p4_registry_types import (
    ContextSurfaceClassification,
    RegisteredContextSurface,
)

_CONTEXT_E2E = "test_txp4_q12_primary_context_model_e2e_attribution"
_CONTEXT_GATE = "test_trace_x_p4_r4_closed_world.py::test_txp4r4_q04_context_surfaces_closed_world_parity"

CONTEXT_SURFACE_REGISTRY: Final[tuple[RegisteredContextSurface, ...]] = (
    RegisteredContextSurface(
        path="intergrax/runtime/nexus/context/context_engine.py",
        surface_kind="call:record_context_assembled_from_engine",
        classification=ContextSurfaceClassification.CANONICAL_CONTEXT_ENGINE_ASSEMBLY,
        semantic_reason="Canonical ContextEngine assembly records CONTEXT_ASSEMBLED via sanctioned recorder",
        canonical_owner="Context Engineering / ContextEngine",
        canonical_contract="record_context_assembled_from_engine",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/context_skill_recording.py",
        surface_kind="def:record_context_assembled_from_engine",
        classification=ContextSurfaceClassification.CANONICAL_CONTEXT_ASSEMBLED_RECORDER,
        semantic_reason="Canonical recorder entrypoint for engine-produced CONTEXT_ASSEMBLED evidence",
        canonical_owner="RuntimeEvent factual evidence",
        canonical_contract="record_context_assembled_from_engine",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/context_skill_recording.py",
        surface_kind="call:bind_pending_context_assembly_event_id",
        classification=ContextSurfaceClassification.CANONICAL_ATTRIBUTION_BIND,
        semantic_reason="Binds pending CONTEXT_ASSEMBLED EventId for P4 model-call attribution",
        canonical_owner="Runtime LLM attribution",
        canonical_contract="bind_pending_context_assembly_event_id",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/context_skill_recording.py",
        surface_kind="construct:context_assembly_payload_v4",
        classification=ContextSurfaceClassification.CANONICAL_CONTEXT_ASSEMBLED_RECORDER,
        semantic_reason="Constructs ContextAssemblyPayloadV4 for canonical CONTEXT_ASSEMBLED emission",
        canonical_owner="RuntimeEvent factual evidence",
        canonical_contract="ContextAssemblyPayloadV4",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/context_skill_recording.py",
        surface_kind="construct:context_assembly_payload_v2",
        classification=ContextSurfaceClassification.CANONICAL_CONTEXT_ASSEMBLED_RECORDER,
        semantic_reason="Constructs ContextAssemblyPayloadV2 for trimmed companion events on assembly path",
        canonical_owner="RuntimeEvent factual evidence",
        canonical_contract="ContextAssemblyPayloadV2",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/context_skill_recording.py",
        surface_kind="emit:runtime_event_context_assembled",
        classification=ContextSurfaceClassification.CANONICAL_CONTEXT_ASSEMBLED_RECORDER,
        semantic_reason="Emits RuntimeEventType.CONTEXT_ASSEMBLED on sanctioned assembly recording path",
        canonical_owner="RuntimeEvent factual evidence",
        canonical_contract="RuntimeEventType.CONTEXT_ASSEMBLED",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/nexus/uaep/uaep_executor.py",
        surface_kind="construct:context_assembly_payload_v2",
        classification=ContextSurfaceClassification.CANONICAL_UAEP_CONTEXT_EMITTER,
        semantic_reason="UAEP legacy assembly path constructs ContextAssemblyPayloadV2 prior to emit",
        canonical_owner="UAEP executor",
        canonical_contract="ContextAssemblyPayloadV2",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/nexus/uaep/uaep_executor.py",
        surface_kind="emit:runtime_event_context_assembled",
        classification=ContextSurfaceClassification.CANONICAL_UAEP_CONTEXT_EMITTER,
        semantic_reason="UAEP executor emits RuntimeEventType.CONTEXT_ASSEMBLED for governed runs",
        canonical_owner="UAEP executor",
        canonical_contract="RuntimeEventType.CONTEXT_ASSEMBLED",
        evidence_nodeid=_CONTEXT_E2E,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/spine_payload_codec.py",
        surface_kind="construct:context_assembly_payload_v4",
        classification=ContextSurfaceClassification.PAYLOAD_CODEC_OR_DESERIALIZE,
        semantic_reason="Spine payload codec reconstructs ContextAssemblyPayloadV4 from stored envelopes (not a producer)",
        canonical_owner="RuntimeEvent payload codec",
        canonical_contract="spine_payload_codec",
        evidence_nodeid=_CONTEXT_GATE,
    ),
    RegisteredContextSurface(
        path="intergrax/runtime/events/spine_payload_codec.py",
        surface_kind="construct:context_assembly_payload_v2",
        classification=ContextSurfaceClassification.PAYLOAD_CODEC_OR_DESERIALIZE,
        semantic_reason="Spine payload codec reconstructs ContextAssemblyPayloadV2 from stored envelopes (not a producer)",
        canonical_owner="RuntimeEvent payload codec",
        canonical_contract="spine_payload_codec",
        evidence_nodeid=_CONTEXT_GATE,
    ),
)
