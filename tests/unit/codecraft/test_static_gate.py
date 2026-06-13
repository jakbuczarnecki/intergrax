# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

import pytest

from intergrax.codecraft.profile import CodeCraftProfile
from intergrax.codecraft.static_gate import StaticCodeGate

pytestmark = pytest.mark.unit


def test_static_gate_passes_safe_code() -> None:
    profile = CodeCraftProfile(mode="autonomous", forbidden_imports=["os"])
    gate = StaticCodeGate(profile)
    result = gate.scan("print('hello')\n")
    assert result.passed is True
    assert result.rule_ids == []


def test_static_gate_blocks_forbidden_import() -> None:
    profile = CodeCraftProfile(mode="autonomous", forbidden_imports=["os"])
    gate = StaticCodeGate(profile)
    result = gate.scan("import os\nprint(os.getcwd())\n")
    assert result.passed is False
    assert "forbidden_import" in result.rule_ids
    assert "os" in result.message


def test_static_gate_blocks_oversize_code() -> None:
    profile = CodeCraftProfile(mode="autonomous", max_code_bytes=256)
    gate = StaticCodeGate(profile)
    result = gate.scan("# " + ("x" * 400) + "\nprint(1)\n")
    assert result.passed is False
    assert "code_size_exceeded" in result.rule_ids


def test_static_gate_blocks_eval_pattern() -> None:
    profile = CodeCraftProfile(mode="autonomous")
    gate = StaticCodeGate(profile)
    result = gate.scan("eval('1+1')\n")
    assert result.passed is False
    assert "forbidden_call_eval" in result.rule_ids


def test_static_gate_blocks_unsupported_language() -> None:
    profile = CodeCraftProfile(mode="autonomous", allowed_languages=["python"])
    gate = StaticCodeGate(profile)
    result = gate.scan("console.log('hi')", language="javascript")
    assert result.passed is False
    assert "language_not_allowed" in result.rule_ids
