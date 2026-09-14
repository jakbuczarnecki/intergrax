# © Artur Czarnecki. All rights reserved.

"""Semantic suite_id label maps for canonical qualification catalog."""

from __future__ import annotations

from collections.abc import Mapping

SHARED_DG001_SUITE_ID = "dg001-lineage"
SHARED_NPSC5D_SUITE_ID = "npsc5d-final"
RUNTIME_EVENTS_SUITE_ID = "runtime-events"
RUNTIME_OBSERVABILITY_SUITE_ID = "runtime-observability"

NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID = "npsc5e-r3-cross-db"

NPSC5E_R3_MANDATORY_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R1 Final": "npsc5e-r3.mandatory.r1-final",
    "R2 Final": "npsc5e-r3.mandatory.r2-final",
    "R3 implementation gate": "npsc5e-r3.mandatory.r3-implementation-gate",
    "P0A": "npsc5e-r3.mandatory.p0a",
    "DG_001": "npsc5e-r3.mandatory.dg-001",
    "NPSC-5A": "npsc5e-r3.mandatory.npsc-5a",
    "NPSC-5B Final": "npsc5e-r3.mandatory.npsc-5b-final",
    "NPSC-5C": "npsc5e-r3.mandatory.npsc-5c",
    "NPSC-5D Final": "npsc5e-r3.mandatory.npsc-5d-final",
    "HITL R3": "npsc5e-r3.mandatory.hitl-r3",
    "Attempt lifecycle": "npsc5e-r3.mandatory.attempt-lifecycle",
    "Child execution": "npsc5e-r3.mandatory.child-execution",
    "Terminal": "npsc5e-r3.mandatory.terminal",
    "Cancellation": "npsc5e-r3.mandatory.cancellation",
    "Checkpoint store": "npsc5e-r3.mandatory.checkpoint-store",
    "Long-running": "npsc5e-r3.mandatory.long-running",
    "Fan-out": "npsc5e-r3.mandatory.fan-out",
}

NPSC5E_R3_EXCLUSIVE_RESOURCE_BY_LABEL: dict[str, str] = {
    "R3 implementation gate": NPSC5E_R3_CROSS_DB_EXCLUSIVE_RESOURCE_ID,
}

NPSC5E_R2_MANDATORY_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R1 Final": "npsc5e-r2.mandatory.r1-final",
    "R2 Original": "npsc5e-r2.mandatory.r2-original",
    "R2-H1": "npsc5e-r2.mandatory.r2-h1",
    "R2-H2": "npsc5e-r2.mandatory.r2-h2",
    "R2-H2-Q1": "npsc5e-r2.mandatory.r2-h2-q1",
    "P0A": "npsc5e-r2.mandatory.p0a",
    "DG_001": SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
    "HITL R3": "npsc5e-r2.mandatory.hitl-r3",
    "NPSC-5A": "npsc5e-r2.mandatory.npsc-5a",
    "NPSC-5B": "npsc5e-r2.mandatory.npsc-5b",
    "NPSC-5C": "npsc5e-r2.mandatory.npsc-5c",
    "Attempt lifecycle": "npsc5e-r2.mandatory.attempt-lifecycle",
    "Child execution": "npsc5e-r2.mandatory.child-execution",
    "Terminal": "npsc5e-r2.mandatory.terminal",
    "Cancellation": "npsc5e-r2.mandatory.cancellation",
    "Checkpoint store": "npsc5e-r2.mandatory.checkpoint-store",
    "Long-running": "npsc5e-r2.mandatory.long-running",
}

NPSC5F_R1_DIRECT_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R1 implementation gate": "npsc5f-r1.implementation-gate",
    "NPSC-5F P0 gate": "npsc5f-r1.p0-gate",
    "Runtime events suites": RUNTIME_EVENTS_SUITE_ID,
    "Runtime observability suites": RUNTIME_OBSERVABILITY_SUITE_ID,
    "DG_001": SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
}

NPSC5F_R2_DIRECT_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R2 implementation gate": "npsc5f-r2.implementation-gate",
    "R1 Final": "npsc5f-r2.mandatory.r1-final-orchestrator",
    "NPSC-5F P0 gate": "npsc5f-r1.p0-gate",
    "Runtime events suites": RUNTIME_EVENTS_SUITE_ID,
    "Runtime observability suites": RUNTIME_OBSERVABILITY_SUITE_ID,
    "TRACE-ASOF": "npsc5f-r2.trace-asof",
    "TRACE-BITEMP": "npsc5f-r2.trace-bitemp",
    "Execution reconstruction": "npsc5f-r2.execution-reconstruction",
    "DG_001": SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
    "NPSC-5B Final": "npsc5e-r3.mandatory.npsc-5b-final",
    "R2 drift classifier": "npsc5f-r2.drift-classifier",
}

NPSC5F_R3_DIRECT_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R3 implementation gate": "npsc5f-r3.implementation-gate",
    "R2 Final": "npsc5f-r3.mandatory.r2-final-orchestrator",
    "R1 Final": "npsc5f-r3.mandatory.r1-final-orchestrator",
    "NPSC-5F P0 gate": "npsc5f-r1.p0-gate",
    "Runtime events suites": RUNTIME_EVENTS_SUITE_ID,
    "Runtime observability suites": RUNTIME_OBSERVABILITY_SUITE_ID,
    "Journal export suites": "npsc5f-r3.journal-export",
    "Export boundary suites": "npsc5f-r3.export-boundary",
    "TRACE-ASOF": "npsc5f-r2.trace-asof",
    "TRACE-BITEMP": "npsc5f-r2.trace-bitemp",
    "Execution reconstruction": "npsc5f-r2.execution-reconstruction",
    "DG_001": SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
    "R3 drift classifier": "npsc5f-r3.drift-classifier",
}

NPSC5E_R1_LABEL_TO_SUITE_ID: dict[str, str] = {
    "NPSC-5E R1 Final": "npsc5e-r1.final",
}

NPSC5E_FINAL_LABEL_TO_SUITE_ID: dict[str, str] = {
    "NPSC-5E recovery plane (R1+R2+R3 finals + section 94)": "npsc5e-final.recovery-plane-orchestrator",
}

NPSC5F_R4_LABEL_TO_SUITE_ID: dict[str, str] = {
    "R4 implementation gate": "npsc5f-r4.implementation-gate",
    "R3 Final": "npsc5f-r4.mandatory.r3-final-orchestrator",
    "R2 Final": "npsc5f-r4.mandatory.r2-final-orchestrator",
    "R1 Final": "npsc5f-r4.mandatory.r1-final-orchestrator",
    "NPSC-5F P0": "npsc5f-r1.p0-gate",
    "TRACE-ASOF": "npsc5f-r2.trace-asof",
    "TRACE-BITEMP": "npsc5f-r4.trace-bitemp",
    "Execution reconstruction": "npsc5f-r2.execution-reconstruction",
    "DG_001": SHARED_DG001_SUITE_ID,
    "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
    "NPSC-5C": "npsc5f-r4.npsc-5c",
    "NPSC-5B Final": "npsc5e-r3.mandatory.npsc-5b-final",
    "NPSC-5A": "npsc5f-r4.npsc-5a",
    "R4 Final drift sentinel": "npsc5f-r4.final-drift-sentinel",
}


def shared_suite_id_overrides() -> Mapping[str, str]:
    return {
        "DG_001": SHARED_DG001_SUITE_ID,
        "NPSC-5D Final": SHARED_NPSC5D_SUITE_ID,
    }
