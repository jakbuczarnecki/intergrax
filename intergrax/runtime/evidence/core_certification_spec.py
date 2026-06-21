# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Core harness certification levels and scenario sets (HEP Band 2ae · EVID-CORE-02/03)."""

from __future__ import annotations

from enum import Enum
from typing import Final

# Stable catalog order (12 scenarios) — certification_report_emitted last in sequence.
CORE_SCENARIO_CATALOG_ORDER: Final[tuple[str, ...]] = (
    "basic_run_completed",
    "trace_persisted",
    "tool_denied_by_policy",
    "high_risk_tool_hitl",
    "budget_exceeded_handled",
    "llm_error_classified",
    "retry_executed",
    "domain_signal_emitted",
    "memory_read_write_recorded",
    "rag_context_event_recorded",
    "cost_report_generated",
    "certification_report_emitted",
)

_CORE_L1_SCENARIOS: Final[frozenset[str]] = frozenset(
    {
        "basic_run_completed",
        "trace_persisted",
        "tool_denied_by_policy",
        "certification_report_emitted",
    }
)

_CORE_L2_SCENARIOS: Final[frozenset[str]] = _CORE_L1_SCENARIOS | frozenset(
    {
        "high_risk_tool_hitl",
        "budget_exceeded_handled",
        "retry_executed",
        "domain_signal_emitted",
    }
)

_CORE_L3_SCENARIOS: Final[frozenset[str]] = _CORE_L2_SCENARIOS | frozenset(
    {
        "llm_error_classified",
        "memory_read_write_recorded",
        "rag_context_event_recorded",
        "cost_report_generated",
    }
)


class CoreCertificationLevel(str, Enum):
    """CORE-L* certification depth for ``intergrax certify core`` (not W-ADAPT L4)."""

    L1 = "CORE-L1"
    L2 = "CORE-L2"
    L3 = "CORE-L3"


class CoreCertificationMode(str, Enum):
    """How certification is invoked — operator-local default for HEP-1."""

    OPERATOR_LOCAL = "operator_local"
    PREFLIGHT_DOCTOR = "preflight_doctor"


class CoreCertificationSurface(str, Enum):
    """Evidence surfaces distinct from certify-core runtime proof."""

    PYTEST_GATE = "pytest_gate"
    DOCTOR_CI = "doctor_ci"
    PHASE_V_CLOSEOUT = "phase_v_closeout"
    W_ADAPT_L4_RUNTIME = "w_adapt_l4_runtime"
    MVP_PROMOTION_GATES = "mvp_promotion_gates"
    CERTIFY_CORE = "certify_core"


CORE_LEVEL_SCENARIOS: Final[dict[CoreCertificationLevel, tuple[str, ...]]] = {
    CoreCertificationLevel.L1: tuple(
        scenario_id for scenario_id in CORE_SCENARIO_CATALOG_ORDER if scenario_id in _CORE_L1_SCENARIOS
    ),
    CoreCertificationLevel.L2: tuple(
        scenario_id for scenario_id in CORE_SCENARIO_CATALOG_ORDER if scenario_id in _CORE_L2_SCENARIOS
    ),
    CoreCertificationLevel.L3: tuple(
        scenario_id for scenario_id in CORE_SCENARIO_CATALOG_ORDER if scenario_id in _CORE_L3_SCENARIOS
    ),
}


_LEVEL_ALIASES: Final[dict[str, CoreCertificationLevel]] = {
    "core-l1": CoreCertificationLevel.L1,
    "l1": CoreCertificationLevel.L1,
    "1": CoreCertificationLevel.L1,
    "core-l2": CoreCertificationLevel.L2,
    "l2": CoreCertificationLevel.L2,
    "2": CoreCertificationLevel.L2,
    "core-l3": CoreCertificationLevel.L3,
    "l3": CoreCertificationLevel.L3,
    "3": CoreCertificationLevel.L3,
}


def normalize_core_level(value: str | CoreCertificationLevel) -> CoreCertificationLevel:
    """Normalize CLI/plan strings to ``CoreCertificationLevel``."""
    if isinstance(value, CoreCertificationLevel):
        return value
    normalized = value.strip()
    if not normalized:
        raise ValueError("core certification level must not be empty")
    upper = normalized.upper()
    try:
        return CoreCertificationLevel(upper)
    except ValueError:
        pass
    alias = _LEVEL_ALIASES.get(normalized.lower())
    if alias is not None:
        return alias
    raise ValueError(f"invalid core certification level: {value!r}")


def scenario_ids_for_level(level: str | CoreCertificationLevel) -> tuple[str, ...]:
    """Return ordered scenario ids required for ``level`` (cumulative L1→L2→L3)."""
    resolved = normalize_core_level(level)
    return CORE_LEVEL_SCENARIOS[resolved]


def is_scenario_in_level(scenario_id: str, level: str | CoreCertificationLevel) -> bool:
    """Return whether ``scenario_id`` is required at ``level`` (cumulative)."""
    resolved = normalize_core_level(level)
    return scenario_id in CORE_LEVEL_SCENARIOS[resolved]
