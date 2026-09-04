# © Artur Czarnecki. All rights reserved.

"""Serialization roundtrip tests for Autonomous Work durable records."""

from __future__ import annotations

import pytest

from intergrax.autonomous_work.serialization import (
    responsibility_from_json,
    responsibility_to_json,
    worker_definition_from_json,
    worker_definition_to_json,
    worker_goal_from_json,
    worker_goal_to_json,
    worker_instance_from_json,
    worker_instance_to_json,
    work_continuity_state_from_json,
    work_continuity_state_to_json,
    worker_principal_binding_from_json,
    worker_principal_binding_to_json,
)
from tests.unit.autonomous_work import repository_contracts as contract_suite

pytestmark = pytest.mark.unit


def test_worker_definition_json_roundtrip() -> None:
    definition = contract_suite.worker_definition()
    assert worker_definition_from_json(worker_definition_to_json(definition)) == definition


def test_worker_instance_json_roundtrip() -> None:
    instance = contract_suite.worker_instance()
    assert worker_instance_from_json(worker_instance_to_json(instance)) == instance


def test_responsibility_json_roundtrip() -> None:
    entity = contract_suite.responsibility()
    assert responsibility_from_json(responsibility_to_json(entity)) == entity


def test_worker_goal_json_roundtrip() -> None:
    entity = contract_suite.worker_goal()
    assert worker_goal_from_json(worker_goal_to_json(entity)) == entity


def test_work_continuity_state_json_roundtrip() -> None:
    entity = contract_suite.continuity_state()
    assert work_continuity_state_from_json(work_continuity_state_to_json(entity)) == entity


def test_worker_principal_binding_json_roundtrip() -> None:
    binding = contract_suite.worker_principal_binding()
    assert (
        worker_principal_binding_from_json(worker_principal_binding_to_json(binding))
        == binding
    )
