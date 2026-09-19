# © Artur Czarnecki. All rights reserved.

"""Explicit HARNESS-01 evidence execution batches (reporting index only)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Harness01EvidenceBatch:
    label: str
    pytest_target: str


HARNESS_01_EVIDENCE_EXECUTION_BATCHES: tuple[Harness01EvidenceBatch, ...] = (
    Harness01EvidenceBatch("HARNESS gates", "tests/qualification/harness_01/"),
    Harness01EvidenceBatch(
        "U3 agent plugin execution closure",
        "tests/unit/runtime/architecture/test_platform_execution_unification_u3_agent_plugin_execution_closure.py",
    ),
    Harness01EvidenceBatch(
        "PLUG-02 public pattern bridge",
        "tests/unit/runtime/nexus/tools/test_plug_02_r1_public_invocation_pattern_evidence.py",
    ),
    Harness01EvidenceBatch("PLUG-03 tool/skill paths", "tests/qualification/plug_03/"),
    Harness01EvidenceBatch("HOST-01 host entry", "tests/qualification/host_01/"),
    Harness01EvidenceBatch("BG-01 background entry", "tests/qualification/bg_01/"),
    Harness01EvidenceBatch(
        "GR10-R8 inner governance",
        "tests/qualification/governance/strategy/test_gr10_r8_orchestration_inner_governance_qualification.py",
    ),
    Harness01EvidenceBatch(
        "GR10 strategy gates",
        "tests/qualification/governance/strategy/test_gr10_gates.py",
    ),
)
