# © Artur Czarnecki. All rights reserved.

"""Composition tests — explicit host governance identity admission wiring."""

from __future__ import annotations

import inspect

import pytest

from intergrax.runtime.execution.environment_host_task_execution import (
    build_environment_host_task_execution,
)
from intergrax.runtime.execution.nexus_host_execution import (
    build_host_task_execution,
    build_host_task_execution as build_nexus_host_task_execution,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_nexus_host_task_execution_builder_requires_admit_dependency() -> None:
    signature = inspect.signature(build_nexus_host_task_execution)
    assert signature.parameters["admit_root_governance_identity"].default is inspect.Parameter.empty
    assert signature.parameters["root_authority_admission"].default is inspect.Parameter.empty


def test_runtime_host_task_execution_builder_requires_admit_dependency() -> None:
    signature = inspect.signature(build_host_task_execution)
    assert signature.parameters["admit_root_governance_identity"].default is inspect.Parameter.empty
    assert signature.parameters["root_authority_admission"].default is inspect.Parameter.empty


def test_runtime_environment_host_task_execution_builder_requires_admit_dependency() -> None:
    signature = inspect.signature(build_environment_host_task_execution)
    assert signature.parameters["admit_root_governance_identity"].default is inspect.Parameter.empty
    assert signature.parameters["root_authority_admission"].default is inspect.Parameter.empty
