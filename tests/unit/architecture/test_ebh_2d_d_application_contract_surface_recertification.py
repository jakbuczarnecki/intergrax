# © Artur Czarnecki. All rights reserved.

"""EBH-2D-D — whole Tier-3 Application contract surface recertification meta-gate."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import AgentFactory, CanonicalAgentFactory
from intergrax.skills.contracts.skill_registry_read import SkillRegistryRead

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax/applications/contracts"

_EBH_2D_PRIOR_GATES = tuple(
    sorted(_REPO_ROOT.glob("tests/unit/architecture/test_ebh_2d_*.py")),
)

_FORBIDDEN_INTERGRAX_PREFIXES = (
    "intergrax.runtime.",
    "intergrax.applications._shared.",
)
_FORBIDDEN_INTERGRAX_EXACT = frozenset(
    {
        "intergrax.agents.agent_contract",
        "intergrax.rag.bootstrap",
        "intergrax.fastapi_core.config",
    }
)
_FORBIDDEN_REGISTRY_SUBSTR = ".registry."

_STD_FORBIDDEN_TOP_LEVEL = frozenset({"importlib", "inspect", "subprocess"})

_ENV_IO_NAMES = frozenset({"environ", "getenv"})

_TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore\b")

# EBH-2D-D outcome A — per-application settings injected by host; not a composition service bag.
_SETTINGS_ANY_ALLOWLIST = frozenset(
    {
        _CONTRACTS_ROOT / "build_context.py",
    }
)

# JSON-wire / migration helpers — declarative normalization only (EBH-2D-B profile ownership).
_ANY_WIRE_MODULE_ALLOWLIST = frozenset(
    {
        "environment_profile/root.py",
        "environment_profile/presets.py",
        "environment_profile/sub_profiles.py",
        "environment_profile/normalization.py",
        "environment_profile/bundles.py",
        "environment_profile/domain_policy.py",
        "environment_profile/decision_profile_legacy.py",
        "environment_state.py",
        "application_migration.py",
        "application_environment_diff.py",
        "runtime_inspection/safe_views.py",
    }
)

_BROAD_CALLABLE_RE = re.compile(r"Callable\[\s*\.\.\.\s*,")


def _discover_contract_modules() -> list[Path]:
    return sorted(
        path
        for path in _CONTRACTS_ROOT.rglob("*.py")
        if path.name != "__pycache__"
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _intergrax_imports(path: Path) -> list[str]:
    tree = ast.parse(_read(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax."):
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.",
        ):
            hits.append(node.module)
    return hits


def _top_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(_read(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _relative_contract(path: Path) -> str:
    return path.relative_to(_CONTRACTS_ROOT).as_posix()


def test_ebh_2d_d_prior_gate_modules_present() -> None:
    assert len(_EBH_2D_PRIOR_GATES) >= 10
    names = {p.name for p in _EBH_2D_PRIOR_GATES}
    for required in (
        "test_ebh_2d_a_application_build_context_boundary.py",
        "test_ebh_2d_b_environment_profile_contract_ownership.py",
        "test_ebh_2d_c_application_contract_ownership.py",
        "test_ebh_2d_c_r6_canonical_agent_routing_decision_closure.py",
    ):
        assert required in names


def test_whole_contract_surface_inventory_minimum_size() -> None:
    modules = _discover_contract_modules()
    assert len(modules) >= 65


def test_whole_surface_forbids_contract_to_implementation_edges() -> None:
    problems: list[str] = []
    for path in _discover_contract_modules():
        for imported in _intergrax_imports(path):
            rel = path.relative_to(_REPO_ROOT)
            if any(imported.startswith(prefix) for prefix in _FORBIDDEN_INTERGRAX_PREFIXES):
                problems.append(f"{rel}: {imported}")
            if _FORBIDDEN_REGISTRY_SUBSTR in f".{imported}.":
                if imported.startswith(
                    (
                        "intergrax.integrations.contracts.",
                        "intergrax.llm_adapters.contracts.",
                        "intergrax.tools.contracts.",
                        "intergrax.skills.contracts.",
                    ),
                ):
                    continue
                problems.append(f"{rel}: {imported}")
            if imported in _FORBIDDEN_INTERGRAX_EXACT or any(
                imported.startswith(f"{name}.") for name in _FORBIDDEN_INTERGRAX_EXACT
            ):
                problems.append(f"{rel}: {imported}")
    assert not problems, "\n".join(problems)


def test_whole_surface_forbids_dynamic_import_and_env_io() -> None:
    problems: list[str] = []
    for path in _discover_contract_modules():
        rel = path.relative_to(_REPO_ROOT)
        roots = _top_level_import_roots(path)
        if not roots.isdisjoint(_STD_FORBIDDEN_TOP_LEVEL):
            problems.append(f"{rel}: forbidden std import {roots & _STD_FORBIDDEN_TOP_LEVEL}")
        tree = ast.parse(_read(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "os" and node.attr in _ENV_IO_NAMES:
                    problems.append(f"{rel}: os.{node.attr}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"__import__", "import_module"}:
                    problems.append(f"{rel}: dynamic import {node.func.id}")
    assert not problems, "\n".join(problems)


def test_whole_surface_has_no_type_ignore() -> None:
    problems: list[str] = []
    for path in _discover_contract_modules():
        if _TYPE_IGNORE_RE.search(_read(path)):
            problems.append(str(path.relative_to(_REPO_ROOT)))
    assert not problems, "\n".join(problems)


def test_application_build_context_settings_any_is_single_allowlisted_site() -> None:
    any_sites: list[str] = []
    for path in _discover_contract_modules():
        source = _read(path)
        if re.search(r"\bsettings:\s*Any\b", source):
            any_sites.append(_relative_contract(path))
    assert any_sites == ["build_context.py"]
    assert _CONTRACTS_ROOT / "build_context.py" in _SETTINGS_ANY_ALLOWLIST
    field = next(f for f in ApplicationBuildContext.__dataclass_fields__.values() if f.name == "settings")
    assert field.type is not None


def test_weak_any_outside_wire_allowlist_is_absent() -> None:
    problems: list[str] = []
    patterns = (
        re.compile(r"\bfrom typing import[^\n]*\bAny\b"),
        re.compile(r"\btyping\.Any\b"),
    )
    for path in _discover_contract_modules():
        rel = _relative_contract(path)
        if rel in _ANY_WIRE_MODULE_ALLOWLIST or rel == "build_context.py":
            continue
        source = _read(path)
        for pattern in patterns:
            if pattern.search(source):
                problems.append(f"{rel}: {pattern.pattern}")
    assert not problems, "\n".join(problems)


def test_public_factory_alias_and_skill_read_typing() -> None:
    assert AgentFactory is CanonicalAgentFactory
    skill_source = _read(_REPO_ROOT / "intergrax/skills/contracts/skill_registry_read.py")
    assert "-> Any" not in skill_source
    assert "Any |" not in skill_source
    assert issubclass(SkillRegistryRead, object)


def test_no_broad_canonical_factory_in_factory_contract() -> None:
    source = _read(_CONTRACTS_ROOT / "factory.py")
    assert "Callable[..., Tier2Agent]" not in source
    assert not _BROAD_CALLABLE_RE.search(source)


def test_contracts_package_does_not_import_shared() -> None:
    init_path = _CONTRACTS_ROOT / "__init__.py"
    for imported in _intergrax_imports(init_path):
        assert not imported.startswith("intergrax.applications._shared.")
