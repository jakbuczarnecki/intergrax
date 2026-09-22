# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R4-R1 — structural LLMAdapter replaceability and declarative metadata resolution."""

from __future__ import annotations

import ast
import importlib
import textwrap
from pathlib import Path

import pytest

from intergrax.applications._shared.agent_resolution import resolve_agent_contract_from_binding
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from model_routing_qualifier.model_routing_qualifier_agent import ModelRoutingQualifierAgent
from tool_selection_qualifier.tool_selection_qualifier_agent import ToolSelectionQualifierAgent
from web_search_qualifier.web_search_qualifier_agent import WebSearchQualifierAgent
from tests.unit.architecture.ebh_2e_external_structural_llm_adapter import (
    ExternalStructuralAdapter,
    assert_external_structural_llm_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENT_RESOLUTION = _REPO_ROOT / "intergrax/applications/_shared/agent_resolution.py"
_QUALIFIER_CONTRACT_IDS = (
    ("web_search_qualifier", WebSearchQualifierAgent),
    ("tool_selection_qualifier", ToolSelectionQualifierAgent),
    ("model_routing_qualifier", ModelRoutingQualifierAgent),
)


def test_ebh_2e_r4_r1_external_adapter_is_runtime_llm_adapter() -> None:
    adapter = ExternalStructuralAdapter()
    assert_external_structural_llm_adapter(adapter)
    assert isinstance(adapter, LLMAdapter)


def test_ebh_2e_r4_r1_external_adapter_does_not_use_base_llm_adapter() -> None:
    fake_path = _REPO_ROOT / "tests/unit/architecture/ebh_2e_external_structural_llm_adapter.py"
    tree = ast.parse(fake_path.read_text(encoding="utf-8"))
    source = fake_path.read_text(encoding="utf-8")
    assert "BaseLLMAdapter" not in source
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "base_llm_adapter" in node.module:
            pytest.fail("ExternalStructuralAdapter must not import BaseLLMAdapter")


@pytest.mark.parametrize("agent_cls", [WebSearchQualifierAgent, ToolSelectionQualifierAgent, ModelRoutingQualifierAgent])
def test_ebh_2e_r4_r1_qualifier_accepts_structural_adapter(agent_cls: type) -> None:
    from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
    from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest

    adapter = ExternalStructuralAdapter()
    assert isinstance(adapter, LLMAdapter)
    agent = agent_cls(llm_adapter=adapter)
    runtime = agent.build_context(
        RuntimeRequest(
            agent_id="qualifier",
            user_id="user",
            session_id="session",
            message="",
            task_id=mint_task_id(),
            run_id=mint_run_id(),
        )
    )
    assert runtime.config.llm_adapter is adapter


def test_ebh_2e_r4_r1_metadata_resolution_has_no_zero_arg_agent_instantiation_token() -> None:
    source = _AGENT_RESOLUTION.read_text(encoding="utf-8")
    assert "agent_type()" not in source
    assert "resolved_agent_type()()" not in source


@pytest.mark.parametrize("contract_id, agent_cls", _QUALIFIER_CONTRACT_IDS)
def test_ebh_2e_r4_r1_qualifier_declarative_contract_resolution(
    contract_id: str,
    agent_cls: type,
) -> None:
    binding = AgentBinding.mount(agent_cls, contract_id=contract_id)
    contract = resolve_agent_contract_from_binding(binding)
    assert contract.id == contract_id
    assert isinstance(contract, AgentContract)


def _install_trap_agent_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> type[Tier2Agent]:
    pkg_root = tmp_path / "metadata_trap_agent_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "contract.py").write_text(
        textwrap.dedent(
            """
            from intergrax.contracts.agent_contract_meta import AgentContract, AgentRiskLevel
            from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState

            def build_agent_contract() -> AgentContract:
                return AgentContract(
                    id="metadata_trap",
                    name="Trap",
                    description="constructor trap",
                    version="0.0.1",
                    capabilities=[],
                    skills=[],
                    extra_tools=[],
                    risk_level=AgentRiskLevel.LOW,
                    lifecycle_state=AgentLifecycleState.DEVELOPMENT,
                    owner_team="test",
                    max_steps=1,
                )
            """
        ),
        encoding="utf-8",
    )
    (pkg_root / "trap_agent.py").write_text(
        textwrap.dedent(
            """
            from intergrax.contracts.agent_contract_meta import AgentContract
            from intergrax.contracts.tier2_agent import Tier2Agent

            class TrapAgent(Tier2Agent):
                def __init__(self, dependency: object) -> None:
                    raise AssertionError("must not instantiate TrapAgent for metadata")

                def get_contract(self) -> AgentContract:
                    raise AssertionError("must not call instance get_contract")
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    module = importlib.import_module("metadata_trap_agent_pkg.trap_agent")
    agent_type = module.TrapAgent
    assert issubclass(agent_type, Tier2Agent)
    return agent_type


def test_ebh_2e_r4_r1_metadata_resolution_does_not_instantiate_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trap_type = _install_trap_agent_package(tmp_path, monkeypatch)
    binding = AgentBinding.mount(trap_type, contract_id="metadata_trap")
    contract = resolve_agent_contract_from_binding(binding)
    assert contract.id == "metadata_trap"


def test_ebh_2e_r4_r1_missing_contract_module_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg_root = tmp_path / "no_contract_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "agent.py").write_text(
        textwrap.dedent(
            """
            from intergrax.contracts.tier2_agent import Tier2Agent

            class NoContractAgent(Tier2Agent):
                def get_contract(self):
                    return None
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    agent_type = importlib.import_module("no_contract_pkg.agent").NoContractAgent
    binding = AgentBinding.mount(agent_type, contract_id="missing")
    with pytest.raises(AgentImportError, match="no declarative contract module"):
        resolve_agent_contract_from_binding(binding)


def test_ebh_2e_r4_r1_broken_contract_import_is_not_masked_as_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg_root = tmp_path / "broken_contract_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "contract.py").write_text(
        "import this_module_does_not_exist_ever\n",
        encoding="utf-8",
    )
    (pkg_root / "agent.py").write_text(
        textwrap.dedent(
            """
            from intergrax.contracts.tier2_agent import Tier2Agent

            class BrokenContractAgent(Tier2Agent):
                def get_contract(self):
                    return None
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    agent_type = importlib.import_module("broken_contract_pkg.agent").BrokenContractAgent
    binding = AgentBinding.mount(agent_type, contract_id="broken")
    with pytest.raises(AgentImportError, match="Failed to import declarative contract module"):
        resolve_agent_contract_from_binding(binding)


def test_ebh_2e_r4_r1_missing_builder_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg_root = tmp_path / "no_builder_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "contract.py").write_text("# no builder\n", encoding="utf-8")
    (pkg_root / "agent.py").write_text(
        textwrap.dedent(
            """
            from intergrax.contracts.tier2_agent import Tier2Agent

            class NoBuilderAgent(Tier2Agent):
                def get_contract(self):
                    return None
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    agent_type = importlib.import_module("no_builder_pkg.agent").NoBuilderAgent
    binding = AgentBinding.mount(agent_type, contract_id="no_builder")
    with pytest.raises(AgentImportError, match="must define callable build_agent_contract"):
        resolve_agent_contract_from_binding(binding)


def test_ebh_2e_r4_r1_wrong_builder_return_type_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pkg_root = tmp_path / "bad_return_pkg"
    pkg_root.mkdir()
    (pkg_root / "__init__.py").write_text("", encoding="utf-8")
    (pkg_root / "contract.py").write_text(
        "def build_agent_contract():\n    return {'not': 'a contract'}\n",
        encoding="utf-8",
    )
    (pkg_root / "agent.py").write_text(
        textwrap.dedent(
            """
            from intergrax.contracts.tier2_agent import Tier2Agent

            class BadReturnAgent(Tier2Agent):
                def get_contract(self):
                    return None
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    agent_type = importlib.import_module("bad_return_pkg.agent").BadReturnAgent
    binding = AgentBinding.mount(agent_type, contract_id="bad_return")
    with pytest.raises(AgentImportError, match="must return AgentContract"):
        resolve_agent_contract_from_binding(binding)
