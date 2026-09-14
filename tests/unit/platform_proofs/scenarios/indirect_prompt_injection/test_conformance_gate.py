# © Artur Czarnecki. All rights reserved.

"""Platform-native architecture gates for indirect_prompt_injection scenario."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    build_order_assistant_lab_manifest,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.tools import SCENARIO_TOOL_IDS
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import (
    WorkflowKind,
    build_scenario_environment_profile,
)
from scripts.proof.scenario_architecture_conformance import (
    ScenarioArchitectureRuleId,
    validate_scenario_application_architecture,
)
pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_APP_DIR = _REPO_ROOT / "platform_proofs" / "scenarios" / "indirect_prompt_injection" / "application"
_SCENARIO_PROOF_DIR = _REPO_ROOT / "platform_proofs" / "scenarios" / "indirect_prompt_injection" / "proof"

_BYPASS_PATTERNS = (
    re.compile(r"conformance_check\s*=\s*is_lab"),
    re.compile(r"allow_custom_tools\s*=\s*True"),
    re.compile(r"skip_catalog_validation\s*=\s*True"),
    re.compile(r"scenario_mode\s*=\s*True"),
)


def _application_sources() -> list[tuple[Path, str]]:
    return [
        (path, path.read_text(encoding="utf-8"))
        for path in sorted(_SCENARIO_APP_DIR.glob("*.py"))
    ]


def test_ipi_application_passes_scenario_architecture_conformance() -> None:
    report = validate_scenario_application_architecture(
        repo_root=_REPO_ROOT,
        scenario_slug="indirect_prompt_injection",
    )
    assert report.skipped is False
    assert not report.violations


def test_ipi_application_passes_agent_lifecycle_conformance_gate() -> None:
    report = validate_scenario_application_architecture(
        repo_root=_REPO_ROOT,
        scenario_slug="indirect_prompt_injection",
    )
    lifecycle_violations = [
        violation
        for violation in report.violations
        if violation.rule_id is ScenarioArchitectureRuleId.AGENT_LIFECYCLE_BYPASS
    ]
    assert lifecycle_violations == []


def test_ipi_application_must_not_disable_conformance_bypasses() -> None:
    violations: list[str] = []
    for path, source in _application_sources():
        rel = path.relative_to(_REPO_ROOT).as_posix()
        for pattern in _BYPASS_PATTERNS:
            if pattern.search(source):
                violations.append(f"{rel} contains forbidden bypass pattern: {pattern.pattern}")
    assert violations == []


def test_ipi_proof_package_does_not_import_application_scenario_control_flow() -> None:
    proof_sources = sorted(_SCENARIO_PROOF_DIR.glob("*.py"))
    violations: list[str] = []
    for path in proof_sources:
        source = path.read_text(encoding="utf-8")
        if "execute_order_assistant_run" in source:
            violations.append(f"{path.relative_to(_REPO_ROOT).as_posix()} invokes application runner")
    assert violations == []


def test_ipi_no_parallel_policy_denial_diagnostic_authority() -> None:
    observability = (_SCENARIO_APP_DIR / "observability.py").read_text(encoding="utf-8")
    order_workflow = (_SCENARIO_APP_DIR / "order_workflow.py").read_text(encoding="utf-8")
    assert "OrderPolicyDenialDiagV1" not in observability
    assert "order_policy_denial" not in order_workflow


def test_ipi_manifest_declares_application_owned_tools() -> None:
    env = build_scenario_environment_profile(WorkflowKind.SAFE_READ)
    manifest = build_order_assistant_lab_manifest(env)
    manifest_tools = {declaration.tool_id for declaration in manifest.application_owned_tools}
    assert manifest_tools == set(SCENARIO_TOOL_IDS)

