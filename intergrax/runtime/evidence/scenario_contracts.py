# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Core certification scenario contracts (HEP Band 2ae · EVID-CORE-03)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.evidence.core_certification_spec import (
    CORE_SCENARIO_CATALOG_ORDER,
    CoreCertificationLevel,
    CORE_LEVEL_SCENARIOS,
    is_scenario_in_level,
    scenario_ids_for_level,
)


class CoreScenarioStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvidenceRefKind(str, Enum):
    RUNTIME_EVENT = "runtime_event"
    TRACE_RECORD = "trace_record"
    POLICY_DECISION = "policy_decision"
    BUDGET_TICK = "budget_tick"
    COST_REPORT = "cost_report"
    CERTIFICATION_REPORT = "certification_report"


class CoreEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: EvidenceRefKind
    ref: str
    description: str = ""


class CoreScenarioExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    expected_signals: list[str] = Field(default_factory=list)


class CoreScenarioContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    title: str
    description: str
    min_level: CoreCertificationLevel
    requires_network: bool = False
    requires_real_llm: bool = False
    deterministic_by_contract: bool = True
    reuse_path: str
    required_evidence_kinds: list[EvidenceRefKind]
    expectation: CoreScenarioExpectation


class CoreScenarioResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    status: CoreScenarioStatus
    evidence_refs: list[CoreEvidenceRef] = Field(default_factory=list)
    message: str = ""


def _contract(
    scenario_id: str,
    title: str,
    description: str,
    min_level: CoreCertificationLevel,
    reuse_path: str,
    required_evidence_kinds: list[EvidenceRefKind],
    expectation_summary: str,
    expected_signals: list[str],
) -> CoreScenarioContract:
    return CoreScenarioContract(
        scenario_id=scenario_id,
        title=title,
        description=description,
        min_level=min_level,
        reuse_path=reuse_path,
        required_evidence_kinds=required_evidence_kinds,
        expectation=CoreScenarioExpectation(
            summary=expectation_summary,
            expected_signals=expected_signals,
        ),
    )


_CORE_SCENARIO_CONTRACT_SPECS: tuple[CoreScenarioContract, ...] = (
    _contract(
        "basic_run_completed",
        "Basic run completed",
        "Echo agent completes a UAEP run with decision and context events.",
        CoreCertificationLevel.L1,
        "tests/integration/agents/test_agent_engine_uaep_echo.py",
        [EvidenceRefKind.RUNTIME_EVENT, EvidenceRefKind.TRACE_RECORD],
        "Agent run reaches COMPLETED with CONTEXT_ASSEMBLED and DECISION_EMITTED.",
        ["CONTEXT_ASSEMBLED", "DECISION_EMITTED", "STEP_STARTED"],
    ),
    _contract(
        "trace_persisted",
        "Trace persisted",
        "Run trace is written and readable from the trace store.",
        CoreCertificationLevel.L1,
        "intergrax/runtime/nexus/tracing/sqlite_trace_store.py",
        [EvidenceRefKind.TRACE_RECORD],
        "Persisted trace readback returns the run journal.",
        ["trace_readback"],
    ),
    _contract(
        "tool_denied_by_policy",
        "Tool denied by policy",
        "Policy engine denies tool access before side effects.",
        CoreCertificationLevel.L1,
        "tests/unit/runtime/kernel/test_step_kernel.py · IDEAL-5.5",
        [EvidenceRefKind.POLICY_DECISION],
        "Policy pre-deny blocks tool execution.",
        ["policy_pre_deny"],
    ),
    _contract(
        "high_risk_tool_hitl",
        "High-risk tool requires HITL",
        "High-risk tool invocation routes through human-in-the-loop policy.",
        CoreCertificationLevel.L2,
        "IDEAL-11.3 HITL policy tests",
        [EvidenceRefKind.POLICY_DECISION, EvidenceRefKind.RUNTIME_EVENT],
        "HIGH risk tool triggers HITL or policy pause.",
        ["require_human_on_critical", "HITL"],
    ),
    _contract(
        "budget_exceeded_handled",
        "Budget exceeded handled",
        "Run budget hard-stop or warn path is exercised.",
        CoreCertificationLevel.L2,
        "IDEAL-24.5 quota hard-stop",
        [EvidenceRefKind.BUDGET_TICK, EvidenceRefKind.COST_REPORT],
        "Budget policy enforces quota when exceeded.",
        ["budget_exceeded", "quota_hard_stop"],
    ),
    _contract(
        "llm_error_classified",
        "LLM error classified",
        "LLM failures are classified without unhandled exceptions.",
        CoreCertificationLevel.L3,
        "REL error handling classification tests",
        [EvidenceRefKind.RUNTIME_EVENT],
        "LLM adapter error maps to classified runtime event.",
        ["llm_error_classified"],
    ),
    _contract(
        "retry_executed",
        "Retry executed",
        "Reliability retry path executes on transient failure.",
        CoreCertificationLevel.L2,
        "REL retry path tests",
        [EvidenceRefKind.RUNTIME_EVENT, EvidenceRefKind.TRACE_RECORD],
        "Retry attempt recorded in trace or event bus.",
        ["retry_executed"],
    ),
    _contract(
        "domain_signal_emitted",
        "Domain signal emitted",
        "Runtime event spine emits catalogued domain signal.",
        CoreCertificationLevel.L2,
        "RuntimeEventBus · event catalog",
        [EvidenceRefKind.RUNTIME_EVENT],
        "Catalogued RuntimeEventType observed on bus.",
        ["RuntimeEvent"],
    ),
    _contract(
        "memory_read_write_recorded",
        "Memory read/write recorded",
        "Memory layer events recorded on run journal.",
        CoreCertificationLevel.L3,
        "MEM wiring events",
        [EvidenceRefKind.RUNTIME_EVENT],
        "Memory read/write surfaces in runtime events or trace.",
        ["memory_read", "memory_write"],
    ),
    _contract(
        "rag_context_event_recorded",
        "RAG context event recorded",
        "RAG or context assembly contributes to CONTEXT_ASSEMBLED.",
        CoreCertificationLevel.L3,
        "RAG/CTX CONTEXT_ASSEMBLED events",
        [EvidenceRefKind.RUNTIME_EVENT],
        "CONTEXT_ASSEMBLED or RAG retrieval event present.",
        ["CONTEXT_ASSEMBLED", "RAG"],
    ),
    _contract(
        "cost_report_generated",
        "Cost report generated",
        "Per-run cost or token attribution artifact available.",
        CoreCertificationLevel.L3,
        "intergrax/context/tracking/assembly_cost.py",
        [EvidenceRefKind.COST_REPORT],
        "Cost or budget report fragment emitted.",
        ["cost_report", "assembly_cost"],
    ),
    _contract(
        "certification_report_emitted",
        "Certification report emitted",
        "Certification run produces structured report artifact.",
        CoreCertificationLevel.L1,
        "build/evidence/core_certification/ (EVID-CORE-05)",
        [EvidenceRefKind.CERTIFICATION_REPORT],
        "JSON/Markdown certification report written after scenarios.",
        ["certification_report"],
    ),
)

CORE_SCENARIO_CONTRACTS: tuple[CoreScenarioContract, ...] = tuple(
    contract
    for scenario_id in CORE_SCENARIO_CATALOG_ORDER
    for contract in _CORE_SCENARIO_CONTRACT_SPECS
    if contract.scenario_id == scenario_id
)


def get_core_scenario_contract(scenario_id: str) -> CoreScenarioContract | None:
    for contract in CORE_SCENARIO_CONTRACTS:
        if contract.scenario_id == scenario_id:
            return contract
    return None


def core_scenario_contracts_for_level(
    level: str | CoreCertificationLevel,
) -> tuple[CoreScenarioContract, ...]:
    ids = scenario_ids_for_level(level)
    by_id = {contract.scenario_id: contract for contract in CORE_SCENARIO_CONTRACTS}
    return tuple(by_id[scenario_id] for scenario_id in ids)


def validate_core_scenario_catalog() -> None:
    """Raise ``ValueError`` when the scenario catalog violates HEP contracts."""
    violations: list[str] = []

    if len(CORE_SCENARIO_CONTRACTS) != 12:
        violations.append(
            f"catalog must contain exactly 12 scenarios, got {len(CORE_SCENARIO_CONTRACTS)}"
        )

    catalog_ids = [contract.scenario_id for contract in CORE_SCENARIO_CONTRACTS]
    if len(catalog_ids) != len(set(catalog_ids)):
        violations.append("catalog contains duplicate scenario ids")

    if list(catalog_ids) != list(CORE_SCENARIO_CATALOG_ORDER):
        violations.append("catalog order must match CORE_SCENARIO_CATALOG_ORDER")

    all_level_ids: set[str] = set()
    for level, scenario_ids in CORE_LEVEL_SCENARIOS.items():
        all_level_ids.update(scenario_ids)

    for scenario_id in all_level_ids:
        if get_core_scenario_contract(scenario_id) is None:
            violations.append(f"CORE_LEVEL_SCENARIOS references missing contract: {scenario_id}")

    for contract in CORE_SCENARIO_CONTRACTS:
        if not is_scenario_in_level(contract.scenario_id, contract.min_level):
            violations.append(
                f"{contract.scenario_id}: min_level {contract.min_level.value} "
                "not reflected in CORE_LEVEL_SCENARIOS"
            )
        if not contract.deterministic_by_contract:
            violations.append(f"{contract.scenario_id}: must be deterministic_by_contract")
        if contract.requires_network:
            violations.append(f"{contract.scenario_id}: must not require network")
        if contract.requires_real_llm:
            violations.append(f"{contract.scenario_id}: must not require real LLM")
        if not contract.required_evidence_kinds:
            violations.append(f"{contract.scenario_id}: required_evidence_kinds must not be empty")

    if not is_scenario_in_level("certification_report_emitted", CoreCertificationLevel.L1):
        violations.append("certification_report_emitted must be in CORE-L1")

    if violations:
        raise ValueError("; ".join(violations))
