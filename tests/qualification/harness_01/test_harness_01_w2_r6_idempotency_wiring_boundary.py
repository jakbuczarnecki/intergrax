# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-W2-R6 — idempotency store wiring must stay Nexus-free."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.qualification.harness_01.nexus_boundary_detector import file_imports_nexus_module

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WIRING = _REPO_ROOT / "intergrax" / "agents" / "persistence" / "idempotency_store_wiring.py"

_PRIVATE_INVOKER_FIELD_MARKERS: tuple[str, ...] = (
    "_pre_effect_coordinator",
    "_executor",
    "_scope_policy",
    "_meaningful_side_effect_authorization",
    "_agent_runtime_governance",
    "_inner_execution_guard",
    "_sandbox_availability",
    "RuntimeToolInvoker",
    "RuntimeContext",
    "build_production_runtime_tool_invoker",
    "resolve_tool_registry",
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_harness_01_w2_r6_idempotency_store_wiring_has_zero_nexus_imports() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    assert not file_imports_nexus_module(source), (
        "idempotency_store_wiring.py must not import intergrax.runtime.nexus"
    )


def test_harness_01_w2_r6_idempotency_store_wiring_has_no_runtime_tool_invoker_private_coupling() -> (
    None
):
    source = _WIRING.read_text(encoding="utf-8")
    offenders = [marker for marker in _PRIVATE_INVOKER_FIELD_MARKERS if marker in source]
    assert offenders == [], (
        "idempotency_store_wiring.py must not reference RuntimeToolInvoker internals:\n"
        + "\n".join(offenders)
    )
