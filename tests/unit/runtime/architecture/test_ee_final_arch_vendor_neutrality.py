# © Artur Czarnecki. All rights reserved.

"""EE-FINAL-ARCH — vendor-neutral authoritative execution core."""

from __future__ import annotations

import pytest

from tests.unit.runtime.architecture._ee_final_arch_facts import (
    ARCH_MODEL,
    scan_vendor_imports_in_execution_core,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_ee_final_arch_provider_abstraction_documented() -> None:
    text = ARCH_MODEL.read_text(encoding="utf-8")
    assert "## 9. Provider abstraction model" in text
    assert "adapter" in text.lower()


def test_ee_final_arch_execution_core_has_zero_vendor_sdk_imports() -> None:
    violations = scan_vendor_imports_in_execution_core()
    assert violations == [], "vendor imports in execution core:\n" + "\n".join(
        violations
    )
