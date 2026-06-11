# © Artur Czarnecki. All rights reserved.

"""Scoreboard generation for Tier-2 agent roster (ACP-PROD-12)."""

from __future__ import annotations

import ast
import importlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
from intergrax.contracts.agent_readiness import (
    DEFAULT_DIMENSION_WEIGHTS,
    AgentProductionReadinessReport,
    AgentProductionReadinessRosterReport,
    AgentReadinessDimension,
    AgentReadinessDimensionScore,
    AgentReadinessStatus,
)
from intergrax.runtime.registry.agent_assembly_resolver import validate_agent_assembly

REPO_ROOT = Path(__file__).resolve().parents[3]
INVENTORY_PATH = REPO_ROOT / "build" / "agent_fleet_inventory.json"
TESTS_ROOT = REPO_ROOT / "tests"

LEGACY_RUNTIME_MARKERS = (
    re.compile(r"\bRuntimeEngine\s*\("),
    re.compile(r"\brun_pipeline_step\s*\("),
    re.compile(r"class\s+\w+\s*\(\s*HarnessReferenceAgent\b"),
)


def load_fleet_inventory(path: Path | None = None) -> dict[str, Any]:
    inventory_path = path or INVENTORY_PATH
    if not inventory_path.is_file():
        raise FileNotFoundError(f"Fleet inventory missing: {inventory_path}")
    return json.loads(inventory_path.read_text(encoding="utf-8"))


def _import_path_from_module(agent_module: str) -> str:
    return agent_module.removeprefix("agents.")


def _agent_class_name(agent_py: Path) -> str | None:
    tree = ast.parse(agent_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Agent"):
            return node.name
    return None


def _load_agent_instance(agent_module: str, agent_py: Path) -> Any:
    module = importlib.import_module(_import_path_from_module(agent_module))
    class_name = _agent_class_name(agent_py)
    if class_name is None:
        raise RuntimeError(f"No Agent class in {agent_py}")
    agent_cls = getattr(module, class_name)
    return agent_cls()


def _inventory_row(inventory: dict[str, Any], agent_id: str) -> dict[str, Any] | None:
    for row in inventory.get("agents", []):
        if row.get("agent_id") == agent_id:
            return row
    return None


def _status_from_pct(pct: float, *, not_applicable: bool = False) -> AgentReadinessStatus:
    if not_applicable:
        return AgentReadinessStatus.NOT_APPLICABLE
    if pct >= 80.0:
        return AgentReadinessStatus.PASS
    if pct >= 40.0:
        return AgentReadinessStatus.PARTIAL
    return AgentReadinessStatus.FAIL


def _dimension(
    dimension: AgentReadinessDimension,
    pct: float,
    *,
    evidence: list[str] | None = None,
    blockers: list[str] | None = None,
    not_applicable: bool = False,
) -> AgentReadinessDimensionScore:
    if not_applicable:
        pct = 0.0
    return AgentReadinessDimensionScore(
        dimension=dimension,
        pct=pct,
        status=_status_from_pct(pct, not_applicable=not_applicable),
        weight=DEFAULT_DIMENSION_WEIGHTS[dimension],
        evidence=list(evidence or []),
        blockers=list(blockers or []),
    )


def _score_contract(contract: AgentContract) -> AgentReadinessDimensionScore:
    result = validate_agent_assembly(contract)
    if result.valid:
        return _dimension(
            AgentReadinessDimension.CONTRACT,
            100.0,
            evidence=["validate_agent_assembly", "ACP-CON-4"],
        )
    return _dimension(
        AgentReadinessDimension.CONTRACT,
        0.0,
        evidence=["validate_agent_assembly"],
        blockers=list(result.errors),
    )


def _score_runtime(row: dict[str, Any] | None, agent_py: Path) -> AgentReadinessDimensionScore:
    text = agent_py.read_text(encoding="utf-8")
    legacy_hits = [pattern.pattern for pattern in LEGACY_RUNTIME_MARKERS if pattern.search(text)]
    if row is None:
        return _dimension(
            AgentReadinessDimension.RUNTIME,
            0.0,
            blockers=["agent not listed in fleet inventory"],
        )
    migrated = row.get("migration_status") == "migrated"
    typed_acp = bool(row.get("typed_acp"))
    if migrated and typed_acp and not legacy_hits:
        return _dimension(
            AgentReadinessDimension.RUNTIME,
            100.0,
            evidence=["ACP-MIG", "cognitive_pattern_base", "check_agent_fleet_migration"],
        )
    blockers: list[str] = []
    if not migrated:
        blockers.append(f"migration_status={row.get('migration_status')}")
    if legacy_hits:
        blockers.extend(legacy_hits)
    pct = 50.0 if migrated else 0.0
    return _dimension(AgentReadinessDimension.RUNTIME, pct, evidence=["audit_agent_fleet_legacy"], blockers=blockers)


def _score_policy() -> AgentReadinessDimensionScore:
    return _dimension(
        AgentReadinessDimension.POLICY,
        85.0,
        evidence=[
            "OrganizationalPolicyEnvelope",
            "merge_organizational_policy_context",
            "org_enforcement kernel",
            "compliance_summary",
        ],
        blockers=["STRICT widen deny on configure_run not agent-specific yet"],
    )


def _score_observability(row: dict[str, Any] | None) -> AgentReadinessDimensionScore:
    if row and row.get("migration_status") == "migrated":
        return _dimension(
            AgentReadinessDimension.OBSERVABILITY,
            80.0,
            evidence=["ACP-OBS-1", "ACP-OBS-2", "AgentRunTrace on typed run"],
        )
    return _dimension(
        AgentReadinessDimension.OBSERVABILITY,
        30.0,
        blockers=["typed ACP run path not verified for agent"],
    )


def _mutating_profile(contract: AgentContract) -> bool:
    if contract.risk_level in {AgentRiskLevel.HIGH, AgentRiskLevel.MEDIUM}:
        return True
    if contract.production_eligible:
        return True
    return contract.id in {"organization_worker", "legal", "local_synthesizer"}


def _score_checkpointing(contract: AgentContract) -> AgentReadinessDimensionScore:
    if not _mutating_profile(contract):
        return _dimension(
            AgentReadinessDimension.CHECKPOINTING,
            0.0,
            evidence=["read-only/staging profile"],
            not_applicable=True,
        )
    return _dimension(
        AgentReadinessDimension.CHECKPOINTING,
        80.0,
        evidence=["AgentCheckpointStore", "ACP-PROD-1", "session resume wiring"],
        blockers=["full crash-recovery acceptance not per-agent yet"],
    )


def _score_idempotency(contract: AgentContract) -> AgentReadinessDimensionScore:
    if not _mutating_profile(contract):
        return _dimension(
            AgentReadinessDimension.IDEMPOTENCY,
            0.0,
            evidence=["read-only/staging profile"],
            not_applicable=True,
        )
    return _dimension(
        AgentReadinessDimension.IDEMPOTENCY,
        80.0,
        evidence=["SideEffectLedger", "ACP-PROD-2", "mutating tool idempotency gate"],
        blockers=["platform idempotency store not wired per host yet"],
    )


def _score_security(agent_py: Path) -> AgentReadinessDimensionScore:
    text = agent_py.read_text(encoding="utf-8")
    violations = [pattern.pattern for pattern in LEGACY_RUNTIME_MARKERS if pattern.search(text)]
    if violations:
        return _dimension(
            AgentReadinessDimension.SECURITY,
            40.0,
            evidence=["check_agent_fleet_migration"],
            blockers=[f"legacy author surface: {item}" for item in violations],
        )
    return _dimension(
        AgentReadinessDimension.SECURITY,
        80.0,
        evidence=["check_agent_step_security.py", "no RuntimeEngine in agent module"],
        blockers=["STRICT profile widen deny not agent-specific yet"],
    )


def _has_eval_tests(agent_id: str, contract_id: str) -> bool:
    needles = {agent_id, contract_id, contract_id.replace("-", "_")}
    for path in TESTS_ROOT.rglob("test_*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            return True
    for path in (REPO_ROOT / "agents").rglob("test_*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if any(needle in text for needle in needles):
            return True
    return False


def _score_evaluation(agent_id: str, contract_id: str) -> AgentReadinessDimensionScore:
    if _has_eval_tests(agent_id, contract_id):
        return _dimension(
            AgentReadinessDimension.EVALUATION,
            90.0,
            evidence=[
                "tests/ or agents/*/tests reference agent",
                "check_agent_release_gates.py",
            ],
        )
    return _dimension(
        AgentReadinessDimension.EVALUATION,
        50.0,
        evidence=["check_agent_release_gates.py baseline"],
        blockers=["no dedicated test reference found for agent"],
    )


def _score_lifecycle(contract: AgentContract) -> AgentReadinessDimensionScore:
    score = 0.0
    evidence: list[str] = []
    blockers: list[str] = []
    if (contract.owner_team or "").strip():
        score += 50.0
        evidence.append("owner_team")
    else:
        blockers.append("owner_team missing")
    if contract.production_eligible:
        if (contract.runbook_ref or "").strip() and (contract.owner_contact or "").strip():
            score += 50.0
            evidence.extend(["runbook_ref", "owner_contact"])
        else:
            blockers.append("production_eligible requires runbook_ref and owner_contact")
    else:
        score += 50.0
        evidence.append("staging lifecycle — runbook optional")
    return _dimension(AgentReadinessDimension.LIFECYCLE, score, evidence=evidence, blockers=blockers)


def _score_capability_routing(agent: Any, contract: AgentContract) -> AgentReadinessDimensionScore:
    if not contract.capabilities:
        return _dimension(
            AgentReadinessDimension.CAPABILITY_ROUTING,
            0.0,
            blockers=["no capabilities on contract"],
        )
    can_handle = getattr(agent, "can_handle", None)
    if not callable(can_handle):
        return _dimension(
            AgentReadinessDimension.CAPABILITY_ROUTING,
            0.0,
            blockers=["can_handle not implemented"],
        )
    return _dimension(
        AgentReadinessDimension.CAPABILITY_ROUTING,
        100.0,
        evidence=[
            "capabilities declared",
            "can_handle implemented",
            "check_capability_routing.py",
            "ACP-CON-6",
        ],
    )


def _overall_pct(dimensions: list[AgentReadinessDimensionScore]) -> float:
    weighted = 0.0
    total_weight = 0.0
    for item in dimensions:
        if item.status == AgentReadinessStatus.NOT_APPLICABLE:
            continue
        weighted += item.pct * item.weight
        total_weight += item.weight
    if total_weight <= 0.0:
        return 0.0
    return round(weighted / total_weight, 2)


def _production_eligible_recommendation(dimensions: list[AgentReadinessDimensionScore], overall: float) -> bool:
    if overall < 90.0:
        return False
    for item in dimensions:
        if item.status == AgentReadinessStatus.NOT_APPLICABLE:
            continue
        if item.pct < 80.0:
            return False
    return True


def build_agent_readiness_report(
    *,
    agent_id: str,
    agent_module: str,
    agent_py: Path,
    inventory: dict[str, Any] | None = None,
    generated_at: datetime | None = None,
) -> AgentProductionReadinessReport:
    inventory = inventory or load_fleet_inventory()
    row = _inventory_row(inventory, agent_id)
    agent = _load_agent_instance(agent_module, agent_py)
    contract = agent.get_contract()
    dimensions = [
        _score_contract(contract),
        _score_runtime(row, agent_py),
        _score_policy(),
        _score_observability(row),
        _score_checkpointing(contract),
        _score_idempotency(contract),
        _score_security(agent_py),
        _score_evaluation(agent_id, contract.id),
        _score_lifecycle(contract),
        _score_capability_routing(agent, contract),
    ]
    overall = _overall_pct(dimensions)
    return AgentProductionReadinessReport(
        agent_id=agent_id,
        contract_id=contract.id,
        generated_at=generated_at or datetime.now(UTC),
        overall_pct=overall,
        production_eligible_recommendation=_production_eligible_recommendation(dimensions, overall),
        dimensions=dimensions,
    )


def _iter_roster_modules(inventory: dict[str, Any]) -> list[tuple[str, str, Path]]:
    modules: list[tuple[str, str, Path]] = []
    for row in inventory.get("agents", []):
        agent_id = str(row["agent_id"])
        module = str(row["agent_module"])
        rel = module.replace(".", "/") + ".py"
        agent_py = REPO_ROOT / rel
        if agent_py.is_file():
            modules.append((agent_id, module, agent_py))
    return modules


def build_roster_readiness_report(
    inventory: dict[str, Any] | None = None,
    *,
    generated_at: datetime | None = None,
) -> AgentProductionReadinessRosterReport:
    inventory = inventory or load_fleet_inventory()
    generated_at = generated_at or datetime.now(UTC)
    reports: list[AgentProductionReadinessReport] = []
    for agent_id, module, agent_py in _iter_roster_modules(inventory):
        reports.append(
            build_agent_readiness_report(
                agent_id=agent_id,
                agent_module=module,
                agent_py=agent_py,
                inventory=inventory,
                generated_at=generated_at,
            )
        )
    overall_values = [report.overall_pct for report in reports]
    runtime_values = [
        report.dimension_score(AgentReadinessDimension.RUNTIME).pct
        for report in reports
        if report.dimension_score(AgentReadinessDimension.RUNTIME) is not None
    ]
    fleet_complete = int(inventory.get("legacy_count", 1)) == 0 and all(
        pct >= 100.0 for pct in runtime_values
    )
    mean_overall = round(sum(overall_values) / len(overall_values), 2) if overall_values else 0.0
    mean_runtime = round(sum(runtime_values) / len(runtime_values), 2) if runtime_values else 0.0
    return AgentProductionReadinessRosterReport(
        generated_at=generated_at,
        agent_count=len(reports),
        roster_mean_overall_pct=mean_overall,
        runtime_dimension_mean_pct=mean_runtime,
        fleet_migration_complete=fleet_complete,
        agents=reports,
    )


def roster_to_markdown(report: AgentProductionReadinessRosterReport) -> str:
    lines = [
        "# Agent Production Readiness Scoreboard",
        "",
        f"Generated: {report.generated_at.isoformat()}",
        f"Agents: {report.agent_count}",
        f"Roster mean overall: {report.roster_mean_overall_pct}%",
        f"Runtime dimension mean: {report.runtime_dimension_mean_pct}%",
        f"Fleet migration complete: {report.fleet_migration_complete}",
        "",
        "| Agent | Contract | Overall % | Runtime % | Prod eligible |",
        "|-------|----------|-----------|-----------|---------------|",
    ]
    for item in report.agents:
        runtime = item.dimension_score(AgentReadinessDimension.RUNTIME)
        runtime_pct = runtime.pct if runtime else 0.0
        lines.append(
            f"| {item.agent_id} | {item.contract_id} | {item.overall_pct} | "
            f"{runtime_pct} | {item.production_eligible_recommendation} |"
        )
    return "\n".join(lines) + "\n"
