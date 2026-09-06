# © Artur Czarnecki. All rights reserved.

"""Typed qualification evidence records."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DockerCrashEvidence:
    kill_method: str
    killed_container_id: str
    killed_exit_code: int | None
    resume_container_id: str
    durable_store_path: str
    window: str
    final_disposition: str


@dataclass(frozen=True, slots=True)
class ScenarioExecutionEvidence:
    scenario_id: str
    invocation: str
    provider: str
    model: str | None
    executed: bool
    decision_path_exercised: bool | None
    used_mock_provider: bool
    block_reason: str | None = None
    decision_id: str | None = None
    outcome: str | None = None
    runtime_modules: frozenset[str] = frozenset()
    requested_provider: str | None = None
    requested_model: str | None = None
    resolved_provider: str | None = None
    resolved_model: str | None = None
    binding_source: str | None = None
