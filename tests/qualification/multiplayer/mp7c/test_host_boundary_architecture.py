# © Artur Czarnecki. All rights reserved.

"""MP-7C — architecture gates for Tier-3 host composition boundary E2E."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    resolve_harness_host_meaningful_side_effect_authorization_wiring,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_MP7C = Path(__file__).resolve().parent
_MP7B = _MP7C.parent / "mp7b"
_LKW_APP = _REPO_ROOT / "applications" / "local_workspace_application"
_QUAL_DOC = (
    _REPO_ROOT
    / "docs"
    / "project"
    / "maintainers"
    / "qualification"
    / "MP-7C_TIER3_HOST_COMPOSITION_BOUNDARY_E2E_QUALIFICATION.md"
)
_WIRING = (
    _REPO_ROOT
    / "intergrax"
    / "applications"
    / "_shared"
    / "harness_meaningful_side_effect_authorization_wiring.py"
)
_CONTRACT_MODULE = (
    _REPO_ROOT / "intergrax" / "contracts" / "meaningful_side_effect_policy.py"
)
_ENFORCEMENT_GATE = (
    _REPO_ROOT / "intergrax" / "collaborative_work" / "enforcement_gate.py"
)
_ORCH_MSE = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_meaningful_side_effect_composition.py"
)
_ORCH_DECISION_BOUND = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_decision_bound_effect_composition.py"
)
_CONSUMER = _MP7B / "consumer.py"
_HOST_COMPOSITION = _MP7C / "host_composition.py"

_LKW_COMPOSITION_ALLOWLIST: frozenset[str] = frozenset()

_FORBIDDEN_PRIVATE_PREFIXES = ("intergrax.collaborative_work",)
_FORBIDDEN_PROVIDER_TOKENS = (
    "PostgreSQLCollaborativeWorkStore",
    "SQLiteCollaborativeWorkStore",
    "InMemoryCollaborativeWorkStore",
)
_FORBIDDEN_REPO_MODULE_SUFFIXES = (
    "collaborative_work.repository",
    "collaborative_work.persistence",
    "collaborative_work.persistence_provider",
    "collaborative_work.in_memory_repository",
    "collaborative_work.enforcement_gate",
)

_STATUS_DOCS = {
    "multiplayer_architecture": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "architecture"
    / "MULTIPLAYER_AI.md",
    "multiplayer_plan": _REPO_ROOT
    / "docs"
    / "project"
    / "capabilities"
    / "plan"
    / "MULTIPLAYER_AI.md",
    "lkw_architecture": _LKW_APP / "docs" / "ARCHITECTURE.md",
    "lkw_plan": _LKW_APP / "docs" / "IMPLEMENTATION_PLAN.md",
}


def _imports_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return found


def _lkw_production_python_files() -> list[Path]:
    if not _LKW_APP.is_dir():
        return []
    skip_parts = {"tests", "docker", "__pycache__", ".proof_docs", "build"}
    return [
        path
        for path in _LKW_APP.rglob("*.py")
        if not any(part in skip_parts for part in path.parts)
    ]


def _rel(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def test_mp7c_qualification_doc_closed() -> None:
    text = _QUAL_DOC.read_text(encoding="utf-8-sig")
    assert "TIER-3 HOST COMPOSITION & BOUNDARY E2E QUALIFIED / CLOSED" in text
    assert "MP-7C-C1" in text
    assert "MP-7C-C1-R1" in text
    assert "CLOSED / CERTIFIED" in text or "CLOSED / RECERTIFIED" in text
    assert "BLOCKING ARCHITECTURE GAPS: NONE" in text
    assert "BLOCKING FINDINGS: NONE" in text
    assert "resolve_harness_host_meaningful_side_effect_authorization_wiring" in text
    assert "LKW PRODUCTION CHANGES = NONE" in text
    assert (
        "PLATFORM OPERATES ON CONTRACTS" in text
        or "contracts, not implementations" in text.lower()
    )
    assert "patch(RuntimePolicyEngine)" in text
    assert "injected runtime policy evaluator" in text.lower() or (
        "injected" in text.lower() and "runtime policy evaluator" in text.lower()
    )
    assert "enforcement_gate implementation module" in text.lower() or (
        "canonical ownership" in text.lower() and "contracts" in text.lower()
    )


def test_mp7c_c1_r1_canonical_evaluator_lives_in_contracts() -> None:
    from intergrax.contracts.meaningful_side_effect_policy import (
        MeaningfulSideEffectPolicyEvaluator,
    )

    assert MeaningfulSideEffectPolicyEvaluator.__module__ == (
        "intergrax.contracts.meaningful_side_effect_policy"
    )
    source = _CONTRACT_MODULE.read_text(encoding="utf-8")
    assert "class MeaningfulSideEffectPolicyEvaluator" in source
    assert "intergrax.collaborative_work" not in source


def test_mp7c_c1_r1_one_production_canonical_definition() -> None:
    """Canonical production Protocol: exactly one definition under intergrax/ source."""
    definitions: list[str] = []
    intergrax_root = _REPO_ROOT / "intergrax"
    for path in intergrax_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "class MeaningfulSideEffectPolicyEvaluator" in text:
            definitions.append(_rel(path))
    assert definitions == [
        "intergrax/contracts/meaningful_side_effect_policy.py",
    ]
    # Vendored docker runtime-context trees are not canonical ownership.
    for app in ("local_workspace_application", "lab_application"):
        vendored = (
            _REPO_ROOT
            / "applications"
            / app
            / "docker"
            / "runtime-context"
            / "intergrax"
            / "collaborative_work"
            / "enforcement_gate.py"
        )
        if vendored.is_file():
            # Stale vendored copies may still name the class; they are not platform SSOT.
            assert vendored.is_relative_to(_REPO_ROOT / "applications" / app / "docker")


def test_mp7c_c1_r1_host_and_runtime_do_not_import_evaluator_from_enforcement_gate() -> None:
    for path in (_WIRING, _ORCH_MSE, _ORCH_DECISION_BOUND):
        mods = _imports_in_file(path)
        assert (
            "MeaningfulSideEffectPolicyEvaluator"
            not in _imported_names_from(
                path, "intergrax.collaborative_work.enforcement_gate"
            )
        )
        assert "intergrax.contracts.meaningful_side_effect_policy" in mods
        assert (
            "MeaningfulSideEffectPolicyEvaluator"
            in _imported_names_from(
                path, "intergrax.contracts.meaningful_side_effect_policy"
            )
        )


def test_mp7c_c1_r1_enforcement_gate_imports_neutral_contract() -> None:
    mods = _imports_in_file(_ENFORCEMENT_GATE)
    assert "intergrax.contracts.meaningful_side_effect_policy" in mods
    assert (
        "MeaningfulSideEffectPolicyEvaluator"
        in _imported_names_from(
            _ENFORCEMENT_GATE, "intergrax.contracts.meaningful_side_effect_policy"
        )
    )
    source = _ENFORCEMENT_GATE.read_text(encoding="utf-8")
    assert "class MeaningfulSideEffectPolicyEvaluator" not in source


def test_mp7c_c1_r1_runtime_and_custom_evaluator_structural_conformance() -> None:
    from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
    from intergrax.contracts.meaningful_side_effect_policy import (
        MeaningfulSideEffectPolicyEvaluator,
    )
    from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
    from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

    class _CustomEvaluator:
        def evaluate_meaningful_side_effect(
            self,
            request: MeaningfulSideEffectRequest,
        ) -> PolicyDecision:
            del request
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason="mp7c-c1-r1-custom",
                policy_rule_id="mp7c.c1.r1.custom",
            )

    assert isinstance(RuntimePolicyEngine(), MeaningfulSideEffectPolicyEvaluator)
    assert isinstance(_CustomEvaluator(), MeaningfulSideEffectPolicyEvaluator)


def _imported_names_from(path: Path, module: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            for alias in node.names:
                names.add(alias.name)
    return names


def test_mp7c_status_markers_in_ssot_docs() -> None:
    required = (
        "MP-7C",
        "TIER-3 HOST COMPOSITION",
        "MP-7 — IN PROGRESS",
        "MP-7B",
    )
    for name, path in _STATUS_DOCS.items():
        text = path.read_text(encoding="utf-8-sig")
        missing = [marker for marker in required if marker not in text]
        assert not missing, f"{name}: missing markers: {missing}"


def test_consumer_imports_only_public_contracts() -> None:
    mods = _imports_in_file(_CONSUMER)
    for mod in mods:
        assert not mod.startswith("intergrax.collaborative_work"), mod
        assert "harness_meaningful_side_effect_authorization_wiring" not in mod
    assert any(m.startswith("intergrax.contracts.") for m in mods)
    source = _CONSUMER.read_text(encoding="utf-8")
    assert "getattr(" not in source
    assert "hasattr(" not in source
    assert "setattr(" not in source
    assert "# type: ignore" not in source
    assert "authorization: Any" not in source
    assert "authorization: object" not in source
    assert "_inner" not in source


def test_mp7c_test_modules_do_not_import_wiring_into_consumer_path() -> None:
    """Consumer module must not import host wiring; composition fixture may."""
    consumer_mods = _imports_in_file(_CONSUMER)
    assert not any(
        "harness_meaningful_side_effect_authorization_wiring" in m
        for m in consumer_mods
    )
    host_mods = _imports_in_file(_HOST_COMPOSITION)
    assert any(
        "harness_meaningful_side_effect_authorization_wiring" in m for m in host_mods
    )


def test_host_composition_fixture_may_import_private_cw() -> None:
    mods = _imports_in_file(_HOST_COMPOSITION)
    assert any(m.startswith("intergrax.collaborative_work") for m in mods)


def test_canonical_resolver_returns_public_port_annotation() -> None:
    sig = inspect.signature(
        resolve_harness_host_meaningful_side_effect_authorization_wiring
    )
    params = sig.parameters
    assert "explicit" in params
    explicit_ann = params["explicit"].annotation
    assert "MeaningfulSideEffectAuthorizationPort" in str(explicit_ann)
    assert "runtime_policy_evaluator" in params
    evaluator_ann = str(params["runtime_policy_evaluator"].annotation)
    assert "MeaningfulSideEffectPolicyEvaluator" in evaluator_ann
    assert "RuntimePolicyEngine" not in evaluator_ann
    # Return type carries authorization_port as public Protocol (via Wiring dataclass).
    source = _WIRING.read_text(encoding="utf-8")
    assert "authorization_port: MeaningfulSideEffectAuthorizationPort | None" in source
    assert "explicit: MeaningfulSideEffectAuthorizationPort | None" in source
    assert (
        "runtime_policy_evaluator: MeaningfulSideEffectPolicyEvaluator | None" in source
    )


def test_wiring_defaults_to_runtime_policy_engine_when_evaluator_absent() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    assert "RuntimePolicyEngine()" in source
    assert "runtime_policy_evaluator" in source
    # Must remain replaceable — no concrete-only public annotation.
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in {
            "resolve_harness_host_meaningful_side_effect_authorization_wiring",
            "resolve_harness_host_meaningful_side_effect_authorization_port",
            "build_harness_host_meaningful_side_effect_authorization_port",
        }:
            for arg in list(node.args.kwonlyargs):
                if arg.arg != "runtime_policy_evaluator":
                    continue
                ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
                assert "MeaningfulSideEffectPolicyEvaluator" in ann
                assert ann.strip() != "RuntimePolicyEngine"


def test_mp7c_allow_fixture_does_not_patch_runtime_policy_engine() -> None:
    source = _HOST_COMPOSITION.read_text(encoding="utf-8")
    assert "_engine_factory" not in source
    assert "runtime_policy_evaluator=" in source
    # Materialization observation patch may remain; semantic engine patch must not.
    assert (
        "harness_meaningful_side_effect_authorization_wiring.RuntimePolicyEngine"
        not in source
    )
    injection_tests = (_MP7C / "test_runtime_policy_evaluator_injection.py").read_text(
        encoding="utf-8",
    )
    assert "runtime_policy_evaluator" in injection_tests


def test_wiring_exposes_protocol_not_concrete_class_in_public_surface() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    # Public return annotations must not name concrete boundary class.
    assert "MeaningfulSideEffectAuthorizationBoundary" not in source
    assert "MeaningfulSideEffectAuthorizationPort" in source


def test_lkw_production_forbids_private_collaborative_work_imports() -> None:
    violations: list[str] = []
    for path in _lkw_production_python_files():
        rel = _rel(path)
        if rel in _LKW_COMPOSITION_ALLOWLIST:
            continue
        for mod in _imports_in_file(path):
            for prefix in _FORBIDDEN_PRIVATE_PREFIXES:
                if mod == prefix or mod.startswith(prefix + "."):
                    violations.append(f"{rel}: import {mod}")
            for suffix in _FORBIDDEN_REPO_MODULE_SUFFIXES:
                if mod == f"intergrax.{suffix}" or mod.endswith(suffix):
                    if f"{rel}: import {mod}" not in violations:
                        violations.append(f"{rel}: import {mod}")
        source = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_PROVIDER_TOKENS:
            if re.search(rf"\b{re.escape(token)}\b", source):
                violations.append(f"{rel}: token {token}")
    assert not violations, "LKW private Multiplayer leakage:\n" + "\n".join(violations)


def test_composition_allowlist_is_explicit_not_broad() -> None:
    for entry in _LKW_COMPOSITION_ALLOWLIST:
        assert entry.startswith("applications/local_workspace_application/")
        assert "*" not in entry
        assert not entry.endswith("/")


def test_no_service_locator_or_global_registry_in_wiring() -> None:
    source = _WIRING.read_text(encoding="utf-8")
    for forbidden in (
        "get_global",
        "GLOBAL_",
        "registry[",
        "service_locator",
        "getattr(",
        "hasattr(",
        "setattr(",
    ):
        assert forbidden not in source


def test_mp7c_qualification_tests_reuse_mp7b_consumer_not_duplicate() -> None:
    consumer_copies = list(_MP7C.glob("**/consumer.py"))
    assert consumer_copies == []
    custom_ports = list(_MP7C.glob("**/custom_ports.py"))
    assert custom_ports == []
    # Test modules import mp7b consumer.
    for path in _MP7C.glob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        if "Tier3MultiplayerConsumer" in text:
            assert "tests.qualification.multiplayer.mp7b.consumer" in text
