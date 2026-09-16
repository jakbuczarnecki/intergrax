# © Artur Czarnecki. All rights reserved.

"""OBS-CONTRACT-BOUNDARY-1-R1/R2 — canonical RuntimeEvent contract ownership and purity gates."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event import (
    RuntimeEvent as LegacyRuntimeEvent,
    RuntimeEventType as LegacyRuntimeEventType,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONTRACT_RUNTIME_EVENT = REPO_ROOT / "intergrax" / "contracts" / "runtime_event.py"
LEGACY_RUNTIME_EVENT = (
    REPO_ROOT / "intergrax" / "runtime" / "events" / "runtime_event.py"
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.obs_diag_conformance]

_FORBIDDEN_CONTRACT_PATTERNS = (
    "register_runtime_event_catalog_enricher",
    "_catalog_enricher",
)


def _run_cold_import(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _explicit_runtime_event_kwargs() -> dict[str, object]:
    from intergrax.contracts.execution_identity import (
        mint_attempt_id,
        mint_event_id,
        mint_execution_id,
        mint_run_id,
        mint_task_id,
    )

    return {
        "event_id": mint_event_id(),
        "tenant_id": "tenant-a",
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "event_type": RuntimeEventType.TOOL_COMPLETED,
        "phase": ExecutionPhase.STEP_EXECUTION,
        "correlation_id": "corr-determinism",
    }


def test_legacy_runtime_event_import_path_is_canonical_contract_type() -> None:
    assert LegacyRuntimeEvent is RuntimeEvent
    assert LegacyRuntimeEventType is RuntimeEventType


def test_runtime_event_model_dump_round_trip_preserves_identity_fields() -> None:
    event = RuntimeEvent(**_explicit_runtime_event_kwargs())
    payload = event.model_dump(mode="json")
    restored = RuntimeEvent.model_validate(payload)
    assert restored.event_id == event.event_id
    assert restored.task_id == event.task_id
    assert restored.run_id == event.run_id
    assert restored.attempt_id == event.attempt_id
    assert restored.execution_id == event.execution_id
    assert restored.event_type == RuntimeEventType.TOOL_COMPLETED
    assert restored.schema_version == "runtime_event.v2"


def test_canonical_runtime_event_module_has_no_catalog_registration_api() -> None:
    source = CONTRACT_RUNTIME_EVENT.read_text(encoding="utf-8")
    for pattern in _FORBIDDEN_CONTRACT_PATTERNS:
        assert pattern not in source


def test_legacy_runtime_event_shim_has_no_import_time_catalog_wiring() -> None:
    source = LEGACY_RUNTIME_EVENT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name != "register_runtime_event_catalog_enricher"


def test_runtime_event_construction_import_order_independence_subprocess() -> None:
    kwargs = _explicit_runtime_event_kwargs()
    serialized_fields = (
        "event_id",
        "tenant_id",
        "task_id",
        "run_id",
        "attempt_id",
        "execution_id",
        "event_type",
        "phase",
        "correlation_id",
        "event_kind",
        "event_category",
        "ops_hint",
        "schema_version",
    )
    identity_keys = (
        "event_id",
        "tenant_id",
        "task_id",
        "run_id",
        "attempt_id",
        "execution_id",
        "correlation_id",
    )
    field_literals = ", ".join(f"{key}={repr(kwargs[key])}" for key in identity_keys)
    compare = ", ".join(f'"{field}": data["{field}"]' for field in serialized_fields)
    script_a = (
        "import json\n"
        "from intergrax.contracts.execution_phase import ExecutionPhase\n"
        "from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType\n"
        f"event = RuntimeEvent({field_literals}, "
        "event_type=RuntimeEventType.TOOL_COMPLETED, "
        "phase=ExecutionPhase.STEP_EXECUTION)\n"
        "data = event.model_dump(mode='json')\n"
        f"print(json.dumps({{{compare}}}))"
    )
    script_b = (
        "import json\n"
        "import intergrax.runtime.events.runtime_event\n"
        "from intergrax.contracts.execution_phase import ExecutionPhase\n"
        "from intergrax.contracts.runtime_event import RuntimeEvent, RuntimeEventType\n"
        f"event = RuntimeEvent({field_literals}, "
        "event_type=RuntimeEventType.TOOL_COMPLETED, "
        "phase=ExecutionPhase.STEP_EXECUTION)\n"
        "data = event.model_dump(mode='json')\n"
        f"print(json.dumps({{{compare}}}))"
    )
    result_a = _run_cold_import(script_a)
    result_b = _run_cold_import(script_b)
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert result_a.stdout.strip() == result_b.stdout.strip()


def test_parse_runtime_event_payload_import_order_independence_subprocess() -> None:
    event = RuntimeEvent(**_explicit_runtime_event_kwargs())
    payload_repr = repr(event.model_dump(mode="json"))
    script_a = (
        "import json\n"
        "from intergrax.contracts.runtime_event import parse_runtime_event_payload\n"
        f"restored = parse_runtime_event_payload({payload_repr})\n"
        "print(restored.event_kind, restored.ops_hint, restored.event_category)"
    )
    script_b = (
        "import intergrax.runtime.events.runtime_event\n"
        "from intergrax.contracts.runtime_event import parse_runtime_event_payload\n"
        f"restored = parse_runtime_event_payload({payload_repr})\n"
        "print(restored.event_kind, restored.ops_hint, restored.event_category)"
    )
    result_a = _run_cold_import(script_a)
    result_b = _run_cold_import(script_b)
    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert result_a.stdout == result_b.stdout
