# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-ADR2 documentation regression gates (architecture freeze only)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_ADR = _REPO / "docs/project/technical/adr/entries/2026-09-20/ADR-HARNESS-001.md"
_README = _REPO / "docs/project/technical/adr/README.md"
_UEA = _REPO / "docs/project/architecture/UNIFIED_EXECUTION_ARCHITECTURE.md"
_HUB = _REPO / "docs/project/maintainers/architecture/EXECUTION_ENGINE.md"


def test_harness_01_adr2_adr_exists_and_accepted() -> None:
    assert _ADR.is_file(), "ADR-HARNESS-001 missing"
    text = _ADR.read_text(encoding="utf-8")
    assert "Accepted (architecture freeze · HARNESS-01-ADR2)" in text
    assert "Nexus = private/internal orchestration implementation inside Execution Engine" in text
    assert "`intergrax/runtime/execution/**`" in text
    assert "`intergrax/runtime/nexus/**`" in text
    assert "HostTaskExecutionPort" in text
    assert "build_execution_engine" in text
    assert "Rejected alternatives" in text
    assert "NexusFacade" in text or "PublicNexusFacade" in text
    assert "DEBT` is not an acceptable final state" in text
    assert "external production Nexus importers = 0" in text
    assert "Resolved by HARNESS-01-ADR2" in text


def test_harness_01_adr2_owner_zones_exclude_full_runtime() -> None:
    text = _ADR.read_text(encoding="utf-8")
    assert "Make all `runtime/*` part of EE" in text
    assert "`intergrax/runtime/task/**`" in text
    assert "EE internal zones are only" in _UEA.read_text(encoding="utf-8") or (
        "runtime/execution/**" in text and "runtime/nexus/**" in text
    )
    # Explicit: task/wiring are not final EE-internal Nexus zones
    assert "Direct Nexus allowed?" in text
    assert "No** (final)" in text or "**No** (final)" in text


def test_harness_01_adr2_public_tool_invocation_surface() -> None:
    text = _ADR.read_text(encoding="utf-8")
    assert "intergrax.tools.invocation_pattern" in text
    assert "ToolInvocationPattern" in text
    assert "ToolInvocationPatternContext" in text
    assert "ToolInvocationInvokerPort" in text
    assert "ToolInvocationPlannerPort" in text
    assert "ToolInvocationPatternResult" in text


def test_harness_01_adr2_indexed_and_crosslinked() -> None:
    readme = _README.read_text(encoding="utf-8")
    assert "ADR-HARNESS-001" in readme
    assert "entries/2026-09-20/ADR-HARNESS-001.md" in readme
    uea = _UEA.read_text(encoding="utf-8")
    assert "ADR-HARNESS-001" in uea
    hub = _HUB.read_text(encoding="utf-8")
    assert "ADR-HARNESS-001" in hub
    assert "HostTaskExecutionPort" in hub


def test_harness_01_adr2_rejects_public_nexus_facade() -> None:
    text = _ADR.read_text(encoding="utf-8")
    for banned in (
        "PublicNexusFacade",
        "NexusService",
        "NexusRuntimeAPI",
        "PublicRuntimeRequest",
    ):
        assert banned in text, f"ADR must explicitly reject {banned}"
