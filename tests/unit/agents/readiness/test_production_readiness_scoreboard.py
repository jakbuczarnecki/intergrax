# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.agents.readiness.scoreboard import (
    build_agent_readiness_report,
    build_roster_readiness_report,
    load_fleet_inventory,
    mutating_checkpoint_idempotency_at_100,
    roster_to_markdown,
)
from intergrax.contracts.agent_readiness import AgentReadinessDimension

REPO_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.unit
@pytest.mark.gate
def test_echo_readiness_runtime_dimension_is_100() -> None:
    inventory = load_fleet_inventory()
    row = next(item for item in inventory["agents"] if item["agent_id"] == "echo")
    report = build_agent_readiness_report(
        agent_id="echo",
        agent_module=row["agent_module"],
        agent_py=REPO_ROOT / "agents" / "echo" / "echo_agent.py",
        inventory=inventory,
    )
    runtime = report.dimension_score(AgentReadinessDimension.RUNTIME)
    assert runtime is not None
    assert runtime.pct == 100.0
    assert report.contract_id == "echo"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.no_ci
def test_roster_scoreboard_fleet_migration_complete() -> None:
    roster = build_roster_readiness_report()
    assert roster.agent_count >= 16
    assert roster.runtime_dimension_mean_pct == 100.0
    assert roster.fleet_migration_complete is True
    markdown = roster_to_markdown(roster)
    assert "echo" in markdown


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.no_ci
def test_scoreboard_json_roundtrip() -> None:
    roster = build_roster_readiness_report()
    payload = json.loads(roster.model_dump_json())
    assert payload["schema_version"] == "agent_production_readiness_roster.v1"
    assert len(payload["agents"]) == roster.agent_count


@pytest.mark.unit
@pytest.mark.gate
def test_mutating_agent_checkpoint_and_idempotency_dimensions_are_100() -> None:
    inventory = load_fleet_inventory()
    row = next(item for item in inventory["agents"] if item["agent_id"] == "organization_worker")
    report = build_agent_readiness_report(
        agent_id="organization_worker",
        agent_module=row["agent_module"],
        agent_py=REPO_ROOT / "agents" / "organization_worker" / "organization_worker_agent.py",
        inventory=inventory,
    )
    checkpoint = report.dimension_score(AgentReadinessDimension.CHECKPOINTING)
    idempotency = report.dimension_score(AgentReadinessDimension.IDEMPOTENCY)
    assert checkpoint is not None
    assert idempotency is not None
    assert checkpoint.pct == 100.0
    assert idempotency.pct == 100.0
    assert not checkpoint.blockers
    assert not idempotency.blockers


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.no_ci
def test_roster_mutating_checkpoint_idempotency_gate() -> None:
    roster = build_roster_readiness_report()
    assert mutating_checkpoint_idempotency_at_100(roster) is True
