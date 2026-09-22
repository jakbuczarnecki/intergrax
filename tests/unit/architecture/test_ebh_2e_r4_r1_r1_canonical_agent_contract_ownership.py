# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R4-R1-R1 — canonical AgentContract ownership and runtime delegation."""

from __future__ import annotations

import ast
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import pytest

from boundary_demo.boundary_demo_agent import BoundaryDemoAgent
from boundary_demo.contract import build_agent_contract as build_boundary_demo_contract
from echo.contract import build_agent_contract as build_echo_contract
from echo.echo_agent import EchoAgent
from intergrax.applications._shared.agent_resolution import resolve_agent_contract_from_binding
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.tier2_agent import Tier2Agent
from lab.contract import build_agent_contract as build_lab_contract
from lab.mock_agents import (
    ComposerMockAgent,
    DocumentMockAgent,
    ResearchMockAgent,
    ValidatorMockAgent,
)
from research.contract import build_agent_contract as build_research_contract
from research.research_agent import ResearchAgent
from research.summary_agent import SummaryAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

AgentT = TypeVar("AgentT", bound=Tier2Agent)

_TRIPLE_CONSISTENCY_CASES: tuple[
    tuple[str, type[Tier2Agent], Callable[[], AgentContract] | Callable[[type[Tier2Agent]], AgentContract]],
    ...,
] = (
    ("echo", EchoAgent, build_echo_contract),
    ("research", ResearchAgent, build_research_contract),
    ("research-summary", SummaryAgent, build_research_contract),
    ("boundary_demo", BoundaryDemoAgent, build_boundary_demo_contract),
    ("lab.research_mock", ResearchMockAgent, build_lab_contract),
    ("lab.document_mock", DocumentMockAgent, build_lab_contract),
    ("lab.validator_mock", ValidatorMockAgent, build_lab_contract),
    ("lab.composer_mock", ComposerMockAgent, build_lab_contract),
)

_RUNTIME_AGENT_MODULES = (
    _REPO_ROOT / "agents/echo/echo_agent.py",
    _REPO_ROOT / "agents/research/research_agent.py",
    _REPO_ROOT / "agents/research/summary_agent.py",
    _REPO_ROOT / "agents/lab/mock_agents.py",
    _REPO_ROOT / "agents/boundary_demo/boundary_demo_agent.py",
)

_CONTRACT_MODULES = (
    _REPO_ROOT / "agents/research/contract.py",
    _REPO_ROOT / "agents/lab/contract.py",
)

_IMPORT_CYCLE_PACKAGES = (
    ("echo", "echo.echo_agent", "echo.contract"),
    ("research", "research.research_agent", "research.contract"),
    ("research_summary", "research.summary_agent", "research.contract"),
    ("lab", "lab.mock_agents", "lab.contract"),
    ("boundary_demo", "boundary_demo.boundary_demo_agent", "boundary_demo.contract"),
)


def _canonical_builder(
    builder: Callable[..., AgentContract],
    agent_cls: type[Tier2Agent],
) -> AgentContract:
    if len(inspect.signature(builder).parameters) == 0:
        return builder()
    return builder(agent_cls)


def _get_contract_def(tree: ast.Module) -> ast.FunctionDef | None:
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "get_contract":
                    return item
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "get_contract":
            return node
    return None


def _function_calls_agent_contract_constructor(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        func_ref = node.func
        if isinstance(func_ref, ast.Name) and func_ref.id == "AgentContract":
            return True
        if isinstance(func_ref, ast.Attribute) and func_ref.attr == "AgentContract":
            return True
    return False


def _function_delegates_to_builder(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "build_agent_contract":
                return True
    return False


@pytest.mark.parametrize("contract_id, agent_cls, builder", _TRIPLE_CONSISTENCY_CASES)
def test_ebh_2e_r4_r1_r1_triple_consistency_builder_runtime_resolver(
    contract_id: str,
    agent_cls: type[Tier2Agent],
    builder: Callable[[], AgentContract] | Callable[[type[Tier2Agent]], AgentContract],
) -> None:
    canonical = _canonical_builder(builder, agent_cls)
    runtime = agent_cls().get_contract()
    binding = AgentBinding.mount(agent_cls, contract_id=contract_id)
    resolved = resolve_agent_contract_from_binding(binding)
    assert runtime == canonical
    assert resolved == canonical


@pytest.mark.parametrize("agent_module", _RUNTIME_AGENT_MODULES)
def test_ebh_2e_r4_r1_r1_runtime_get_contract_does_not_construct_agent_contract(
    agent_module: Path,
) -> None:
    tree = ast.parse(agent_module.read_text(encoding="utf-8"))
    get_contract = _get_contract_def(tree)
    assert get_contract is not None, f"missing get_contract in {agent_module}"
    assert not _function_calls_agent_contract_constructor(get_contract), (
        f"{agent_module} must not construct AgentContract inside get_contract"
    )
    assert _function_delegates_to_builder(get_contract), (
        f"{agent_module} get_contract must delegate to build_agent_contract"
    )


@pytest.mark.parametrize("contract_module", _CONTRACT_MODULES)
def test_ebh_2e_r4_r1_r1_contract_module_does_not_import_runtime_agents(
    contract_module: Path,
) -> None:
    source = contract_module.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_suffixes = (
        "research_agent",
        "summary_agent",
        "mock_agents",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for suffix in forbidden_suffixes:
                assert suffix not in node.module, (
                    f"{contract_module} must not import runtime agent modules ({node.module})"
                )


@pytest.mark.parametrize("label, agent_mod, contract_mod", _IMPORT_CYCLE_PACKAGES)
def test_ebh_2e_r4_r1_r1_clean_import_agent_then_contract(
    label: str,
    agent_mod: str,
    contract_mod: str,
) -> None:
    importlib.invalidate_caches()
    importlib.import_module(agent_mod)
    importlib.import_module(contract_mod)


@pytest.mark.parametrize("label, agent_mod, contract_mod", _IMPORT_CYCLE_PACKAGES)
def test_ebh_2e_r4_r1_r1_clean_import_contract_then_agent(
    label: str,
    agent_mod: str,
    contract_mod: str,
) -> None:
    importlib.invalidate_caches()
    importlib.import_module(contract_mod)
    importlib.import_module(agent_mod)
