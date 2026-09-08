# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from intergrax.applications._shared.interaction_wiring import wire_interaction_intake_service
from intergrax.applications._shared.task_control_wiring import TaskEnricher

pytestmark = pytest.mark.unit


def test_wire_interaction_intake_service_signature_has_no_nexus_loop() -> None:
    signature = inspect.signature(wire_interaction_intake_service)
    assert "nexus_loop" not in signature.parameters


def test_wire_interaction_intake_service_uses_typed_task_enricher() -> None:
    annotation = inspect.signature(wire_interaction_intake_service).parameters["task_enricher"].annotation
    assert annotation in {TaskEnricher | None, "TaskEnricher | None"}


def test_interaction_wiring_source_has_no_nexus_tokens() -> None:
    source_path = Path(inspect.getfile(wire_interaction_intake_service))
    source = source_path.read_text(encoding="utf-8")
    assert "NexusLoop" not in source
    assert "Callable[..., object]" not in source
