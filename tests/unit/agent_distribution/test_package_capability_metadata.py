# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.agent_distribution.agent_capability_metadata import (
    AgentCapabilityDescriptor,
    AgentCapabilityDescriptorConflictError,
)
from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
    AgentProjectMetadata,
    AgentProjectMetadataParseError,
    parse_agent_project_pyproject,
    project_agent_capability_descriptors,
)
from intergrax.agent_distribution.builtin_capability_metadata import (
    PackageAgentCapabilityMetadataProvider,
)
from intergrax.runtime.architecture.capability_graph import (
    CapabilityNodeType,
    build_catalog_capability_graph,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILTIN_PROVIDER_MODULE = (
    REPO_ROOT / "intergrax" / "agent_distribution" / "builtin_capability_metadata.py"
)
CAPABILITY_GRAPH_MODULE = (
    REPO_ROOT / "intergrax" / "runtime" / "architecture" / "capability_graph.py"
)

_FORBIDDEN_PROVIDER_IDENTITIES = frozenset(
    {"echo", "legal", "research", "research-summary"},
)
_FORBIDDEN_EXECUTABLE_FRAGMENTS = (
    "AgentRegistry",
    "get_contract()",
    "importlib.import_module",
    "echo.echo_agent",
    "EchoAgent",
    "legal.legal_agent",
    "research.research_agent",
)


def _write_package(
    root: Path,
    *,
    name: str,
    version: str,
    contracts: str,
) -> Path:
    package_dir = root / name
    package_dir.mkdir()
    (package_dir / "pyproject.toml").write_text(
        (
            f"[project]\nname = {name!r}\nversion = {version!r}\n\n"
            f"{contracts}"
        ),
        encoding="utf-8",
    )
    return package_dir


def test_parse_agent_project_pyproject_projects_descriptor_fields() -> None:
    metadata = parse_agent_project_pyproject(
        """
[project]
name = "intergrax-sample-agent"
version = "4.5.6"

[[tool.intergrax.agent.contracts]]
contract_id = "sample-contract"
contract_version = "4.5.6"
capabilities = ["sample.cap"]
skill_ids = ["sample.skill"]
tool_ids = ["sample.tool"]
"""
    )
    assert metadata.distribution_package_id == "intergrax-sample-agent"
    assert metadata.package_version == "4.5.6"
    descriptors = project_agent_capability_descriptors(metadata)
    assert descriptors == (
        AgentCapabilityDescriptor(
            contract_id="sample-contract",
            agent_version="4.5.6",
            capabilities=("sample.cap",),
            skill_ids=("sample.skill",),
            tool_ids=("sample.tool",),
        ),
    )


def test_declared_contracts_without_contract_version_fail_closed() -> None:
    with pytest.raises(AgentProjectMetadataParseError, match="missing contract_version"):
        parse_agent_project_pyproject(
            """
[project]
name = "intergrax-sample-agent"
version = "1.0.0"

[[tool.intergrax.agent.contracts]]
contract_id = "sample-contract"
"""
        )


def test_declared_contracts_without_project_version_fail_closed() -> None:
    with pytest.raises(AgentProjectMetadataParseError, match="no synthetic version fallback"):
        parse_agent_project_pyproject(
            """
[project]
name = "intergrax-sample-agent"

[[tool.intergrax.agent.contracts]]
contract_id = "sample-contract"
contract_version = "2.0.0"
"""
        )


def test_project_agent_capability_descriptors_uses_contract_version_not_package_version() -> None:
    metadata = AgentProjectMetadata(
        distribution_package_id="intergrax-sample-agent",
        package_version="9.9.9",
        declared_contracts=(
            AgentPackageContractDeclaration(
                contract_id="sample-contract",
                contract_version="2.0.0",
            ),
        ),
    )
    descriptors = project_agent_capability_descriptors(metadata)
    assert descriptors[0].agent_version == "2.0.0"


def test_package_provider_preserves_exact_version_in_capability_graph(tmp_path: Path) -> None:
    package_dir = _write_package(
        tmp_path,
        name="versioned-agent",
        version="9.8.7",
        contracts=(
            "[[tool.intergrax.agent.contracts]]\n"
            'contract_id = "versioned-contract"\n'
            'contract_version = "9.8.7"\n'
            'capabilities = ["versioned.cap"]\n'
            'skill_ids = ["harness.tool_smoke"]\n'
            'tool_ids = ["rag.retrieve"]\n'
        ),
    )
    provider = PackageAgentCapabilityMetadataProvider(package_roots=(package_dir,))
    graph = build_catalog_capability_graph(agent_metadata_provider=provider)
    node = next(item for item in graph.nodes if item.node_id == "agent:versioned-contract")
    assert node.version == "9.8.7"
    assert node.metadata["capabilities"] == "versioned.cap"
    edge_keys = {(edge.source_node_id, edge.target_node_id) for edge in graph.edges}
    assert ("agent:versioned-contract", "skill:harness.tool_smoke") in edge_keys
    assert ("agent:versioned-contract", "tool:rag.retrieve") in edge_keys


def test_new_package_metadata_requires_no_platform_core_edit(tmp_path: Path) -> None:
    package_dir = _write_package(
        tmp_path,
        name="external-agent",
        version="2.0.0",
        contracts=(
            "[[tool.intergrax.agent.contracts]]\n"
            'contract_id = "external-new"\n'
            'contract_version = "2.0.0"\n'
        ),
    )
    provider = PackageAgentCapabilityMetadataProvider(package_roots=(package_dir,))
    graph = build_catalog_capability_graph(agent_metadata_provider=provider)
    assert any(node.node_id == "agent:external-new" for node in graph.nodes)


def test_conflicting_package_contract_descriptors_fail_closed(tmp_path: Path) -> None:
    left = _write_package(
        tmp_path,
        name="left-agent",
        version="1.0.0",
        contracts='[[tool.intergrax.agent.contracts]]\ncontract_id = "shared-id"\ncontract_version = "1.0.0"\n',
    )
    right = _write_package(
        tmp_path,
        name="right-agent",
        version="2.0.0",
        contracts='[[tool.intergrax.agent.contracts]]\ncontract_id = "shared-id"\ncontract_version = "2.0.0"\n',
    )
    provider = PackageAgentCapabilityMetadataProvider(package_roots=(left, right))
    with pytest.raises(AgentCapabilityDescriptorConflictError, match="conflicting"):
        provider.list_agent_capability_descriptors()


def test_representative_package_contract_versions_match_runtime_not_package_version() -> None:
    provider = PackageAgentCapabilityMetadataProvider(
        package_roots=(
            REPO_ROOT / "agents" / "echo",
            REPO_ROOT / "agents" / "legal",
            REPO_ROOT / "agents" / "research",
        )
    )
    descriptors = {
        item.contract_id: item for item in provider.list_agent_capability_descriptors()
    }
    assert descriptors["echo"].agent_version == "1.0.0"
    assert descriptors["echo"].capabilities == ("echo.basic",)
    assert descriptors["echo"].skill_ids == ("harness.tool_smoke",)
    assert descriptors["legal"].agent_version == "0.1.0"
    assert descriptors["legal"].capabilities == ("legal.review",)
    assert descriptors["research"].agent_version == "0.1.0"
    assert descriptors["research"].capabilities == ("research.web_search", "research.pipeline")
    assert descriptors["research-summary"].agent_version == "0.1.0"
    assert descriptors["research-summary"].capabilities == ("research.summarize",)


def test_multi_contract_package_preserves_distinct_contract_versions(tmp_path: Path) -> None:
    package_dir = _write_package(
        tmp_path,
        name="multi-contract-agent",
        version="0.1.0",
        contracts=(
            "[[tool.intergrax.agent.contracts]]\n"
            'contract_id = "contract-a"\n'
            'contract_version = "3.4.5"\n'
            'capabilities = ["alpha.cap"]\n'
            "\n"
            "[[tool.intergrax.agent.contracts]]\n"
            'contract_id = "contract-b"\n'
            'contract_version = "6.7.8"\n'
            'capabilities = ["beta.cap"]\n'
        ),
    )
    provider = PackageAgentCapabilityMetadataProvider(package_roots=(package_dir,))
    descriptors = {
        item.contract_id: item for item in provider.list_agent_capability_descriptors()
    }
    assert descriptors["contract-a"].agent_version == "3.4.5"
    assert descriptors["contract-b"].agent_version == "6.7.8"
    assert descriptors["contract-a"].agent_version != descriptors["contract-b"].agent_version


def test_empty_package_provider_does_not_seed_known_inventory() -> None:
    provider = PackageAgentCapabilityMetadataProvider()
    assert provider.list_agent_capability_descriptors() == ()
    graph = build_catalog_capability_graph()
    agent_ids = {
        node.node_id
        for node in graph.nodes
        if node.node_type == CapabilityNodeType.AGENT
    }
    assert not agent_ids


def test_builtin_provider_module_has_no_embedded_agent_identities() -> None:
    source = BUILTIN_PROVIDER_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_constants = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    embedded = sorted(value for value in string_constants if value in _FORBIDDEN_PROVIDER_IDENTITIES)
    assert not embedded, f"embedded agent identities: {embedded}"

    descriptor_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "AgentCapabilityDescriptor")
            or (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "AgentCapabilityDescriptor"
            )
        )
    ]
    assert not descriptor_calls, "generic provider must not construct descriptor inventory"


def test_capability_metadata_path_does_not_import_or_instantiate_agents() -> None:
    for path in (BUILTIN_PROVIDER_MODULE, CAPABILITY_GRAPH_MODULE):
        text = path.read_text(encoding="utf-8")
        violations = [
            f"{path.name}: contains forbidden fragment {fragment!r}"
            for fragment in _FORBIDDEN_EXECUTABLE_FRAGMENTS
            if fragment in text
        ]
        assert not violations, "\n".join(violations)
