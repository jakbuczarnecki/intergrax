# © Artur Czarnecki. All rights reserved.

"""AW-1A — Autonomous Work architecture gate / negative contract tests."""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import fields

import pytest

from intergrax.contracts.autonomous_work import (
    Responsibility,
    WorkContinuityState,
    WorkerDefinition,
    WorkerGoal,
    WorkerInstance,
)


@pytest.mark.unit
def test_worker_definition_has_no_embedded_memory_configuration() -> None:
    field_names = {field.name for field in fields(WorkerDefinition)}
    forbidden = {
        "memory_config",
        "memory_settings",
        "context_window",
        "embedding_config",
        "retrieval_config",
    }
    assert forbidden.isdisjoint(field_names)
    assert "memory_profile_ref" in field_names


@pytest.mark.unit
def test_worker_instance_has_no_credentials() -> None:
    field_names = {field.name for field in fields(WorkerInstance)}
    forbidden = {"credentials", "api_key", "secret", "token", "password"}
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_responsibility_and_goal_have_no_permissions() -> None:
    responsibility_fields = {field.name for field in fields(Responsibility)}
    goal_fields = {field.name for field in fields(WorkerGoal)}
    forbidden = {"permissions", "authority", "authority_scopes", "credentials"}
    assert forbidden.isdisjoint(responsibility_fields)
    assert forbidden.isdisjoint(goal_fields)


@pytest.mark.unit
def test_work_continuity_state_has_no_prompt_or_history_blob() -> None:
    field_names = {field.name for field in fields(WorkContinuityState)}
    forbidden = {
        "conversation",
        "conversation_history",
        "chat_history",
        "prompt",
        "prompt_blob",
        "execution_trace",
        "trace_blob",
        "raw_snapshot",
    }
    assert forbidden.isdisjoint(field_names)


@pytest.mark.unit
def test_autonomous_work_contracts_do_not_import_runtime_services() -> None:
    package = importlib.import_module("intergrax.contracts.autonomous_work")
    module_names = [
        module_info.name
        for module_info in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        )
    ]
    forbidden_prefixes = (
        "intergrax.runtime",
        "intergrax.applications",
        "agents.",
        "applications.",
    )
    for module_name in module_names:
        module = importlib.import_module(module_name)
        source_path = getattr(module, "__file__", "") or ""
        assert source_path
        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()
        for line in source.splitlines():
            stripped = line.strip()
            if not stripped.startswith("from ") and not stripped.startswith("import "):
                continue
            for prefix in forbidden_prefixes:
                assert prefix not in stripped, (
                    f"{module_name} imports forbidden runtime: {line}"
                )
