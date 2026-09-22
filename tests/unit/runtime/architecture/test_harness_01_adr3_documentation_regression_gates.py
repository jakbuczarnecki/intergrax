# © Artur Czarnecki. All rights reserved.

"""HARNESS-01-R5-ADR3 documentation regression gates (architecture freeze only)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_ADR = _REPO / "docs/project/technical/adr/entries/2026-09-22/ADR-HARNESS-003.md"
_README = _REPO / "docs/project/technical/adr/README.md"
_HUB = _REPO / "docs/project/maintainers/architecture/EXECUTION_ENGINE.md"


def test_harness_01_adr3_adr_exists_and_accepted() -> None:
    assert _ADR.is_file(), "ADR-HARNESS-003 missing"
    text = _ADR.read_text(encoding="utf-8")
    assert "Accepted (architecture freeze · HARNESS-01-R5-ADR3)" in text
    assert "HARNESS-01-R5-ADR3" in text
    assert "EBCI-01" in text and "EBCI-12" in text
    assert "PER-CALL IMMUTABLE" in text
    assert "bind_execution_identity" in text
    assert "DEPRECATE" in text
    assert "LEGAL INTERNAL PROJECTION" in text
    assert "second `ExecutionRuntime`" in text
    assert "UniversalExecutionBoundInvoker" in text or "UniversalToolInvoker" in text
    assert "ToolInvocationInvokerPort" in text
    assert "ExecutionBoundCatalogToolInvoker" in text
    assert "ExecutionBoundDeclarativeToolInvoker" in text
    assert "CompensationSideEffectExecutionPort" in text
    assert "RuntimeToolInvoker" in text
    assert "ADR-HARNESS-001" in text
    assert "does not supersede D5" in text or "Does not silently supersede ADR-HARNESS-001 D5" in text


def test_harness_01_adr3_three_layer_and_variant_b() -> None:
    text = _ADR.read_text(encoding="utf-8")
    assert "Three-layer invocation model" in text
    assert "**Accepted**" in text
    assert "Variant" in text
    assert "Rejected" in text


def test_harness_01_adr3_indexed_and_crosslinked() -> None:
    readme = _README.read_text(encoding="utf-8")
    assert "ADR-HARNESS-003" in readme
    assert "entries/2026-09-22/ADR-HARNESS-003.md" in readme
    hub = _HUB.read_text(encoding="utf-8")
    assert "ADR-HARNESS-003" in hub
