# © Artur Czarnecki. All rights reserved.

"""ME-14-C2 canonical tool execution wiring closure gates and proofs."""

from __future__ import annotations

import ast
import asyncio
import importlib
from pathlib import Path

import pytest

from intergrax.runtime.task.task import TaskState
from intergrax.tools.errors import DynamicToolAcquisitionResolutionError
from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_DIGEST_V2,
    ME14_OUTPUT_V1,
    ME14_OUTPUT_V2,
    ME14_PACKAGE_REFERENCE_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V1,
    ME14_VERSION_V2,
)
from testing_support.marketplace_tool_execution_composition import (
    MarketplaceToolExecutionProofStack,
    me14_default_listing_v1,
    me14_listing_v2,
)
from testing_support.me14_tool_catalog_provider import ME14_CATALOG_SOURCE_ID
from testing_support.me14_tool_harness_execution import (
    Me14ToolProofAgent,
    execute_me14_tool_via_host_execution_engine,
    me14_tool_proof_environment,
    me14_tool_proof_manifest,
    run_me14_tool_host_execution,
)
from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
from intergrax.tools.catalog import ToolCatalogProviderRegistry
from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest, DynamicToolAcquisitionService
from intergrax.tools.host_lifecycle import ToolHostLifecycleService
from testing_support.me14_tool_activation_materializer import Me14ToolHostActivationMaterializer
from testing_support.me14_tool_catalog_provider import Me14ToolCatalogProvider
from intergrax.contracts.capability_catalog import (
    CapabilityDiscoveryIdentity,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
)
from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
from testing_support.builder import FakeLLMAdapter

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_ME14_PRIMARY_PATHS = (
    Path("testing_support/me14_tool_harness_execution.py"),
    Path("testing_support/marketplace_tool_execution_composition.py"),
)
_FORBIDDEN_PRIVATE_ATTRS = (
    "_internal_composition",
    "_orchestration_backend",
    "_declarative_tool_invoker",
)
_FORBIDDEN_MANUAL_WIRING = (
    "inject_acp_tool_invoker_metadata",
    "RuntimeToolInvoker",
    "RegistryToolExecutor",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_primary_sources() -> str:
    root = _repo_root()
    return "\n".join(path.read_text(encoding="utf-8") for path in _ME14_PRIMARY_PATHS)


def _ast_names_in_module(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    source = Path(module.__file__).resolve().read_text(encoding="utf-8")
    tree = ast.parse(source)
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def test_me14_c2_execution_uses_only_public_host_runtime_wiring(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path / "lifecycle")
    tool_id, output, task_id = asyncio.run(
        execute_me14_tool_via_host_execution_engine(
        registry=stack.lifecycle.registry_read(),
        tool_logical_id=ME14_TOOL_LOGICAL_ID,
            tmp_path=tmp_path / "exec",
        ),
    )
    assert tool_id == ME14_TOOL_LOGICAL_ID
    assert output == ME14_OUTPUT_V1
    assert task_id is not None


def test_me14_c2_application_tool_registry_is_resolved_by_execution_composition(
    tmp_path: Path,
) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path / "lifecycle")
    activated = stack.lifecycle.registry_read()
    manifest = me14_tool_proof_manifest(ME14_TOOL_LOGICAL_ID)
    environment = me14_tool_proof_environment(ME14_TOOL_LOGICAL_ID)
    host_runtime = build_harness_host_runtime(
        manifest,
        environment,
        tenant_id="tenant-me14",
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "runtime_events.db",
        application_tool_registry=activated,
        llm_adapter=FakeLLMAdapter(),
    )
    wired_registry = host_runtime.env_wiring.tool_wiring.registry
    assert wired_registry.has(ME14_TOOL_LOGICAL_ID)
    empty_stack = MarketplaceToolExecutionProofStack.build()
    empty_registry = empty_stack.lifecycle.registry_read()
    assert not empty_registry.has(ME14_TOOL_LOGICAL_ID)
    from intergrax.applications.contracts.application_package import (
        ApplicationPackageClosureError,
    )

    with pytest.raises(ApplicationPackageClosureError, match="missing from wired tool registry"):
        asyncio.run(
            run_me14_tool_host_execution(
                registry=empty_registry,
                tool_logical_id=ME14_TOOL_LOGICAL_ID,
                tmp_path=tmp_path / "empty",
            ),
        )


def test_me14_c2_primary_composition_has_no_private_execution_attribute_access() -> None:
    combined = _read_primary_sources()
    for forbidden in _FORBIDDEN_PRIVATE_ATTRS:
        assert forbidden not in combined


def test_me14_c2_proof_agent_does_not_install_custom_runtime_tool_invoker() -> None:
    names = _ast_names_in_module("testing_support.me14_tool_harness_execution")
    assert "_execution_registry" not in names
    for forbidden in _FORBIDDEN_MANUAL_WIRING:
        assert forbidden not in names
    tree = ast.parse(
        Path(
            importlib.import_module("testing_support.me14_tool_harness_execution").__file__,
        ).read_text(encoding="utf-8"),
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "tool_invoker":
            raise AssertionError("proof harness must not assign or read config.tool_invoker")


def test_me14_c2_tool_executes_after_lifecycle_without_manual_metadata_injection(
    tmp_path: Path,
) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    evidence = stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path)
    assert evidence.execution_result == ME14_OUTPUT_V1
    assert "inject_acp_tool_invoker_metadata" not in _read_primary_sources()


def test_me14_c2_execution_fails_before_tool_activation(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    from intergrax.applications.contracts.application_package import (
        ApplicationPackageClosureError,
    )

    with pytest.raises(ApplicationPackageClosureError, match="missing from wired tool registry"):
        asyncio.run(
            run_me14_tool_host_execution(
                registry=stack.lifecycle.registry_read(),
                tool_logical_id=ME14_TOOL_LOGICAL_ID,
                tmp_path=tmp_path,
            ),
        )


def test_me14_c2_execution_succeeds_after_tool_activation(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path / "handoff")
    tool_id, output, _ = asyncio.run(
        execute_me14_tool_via_host_execution_engine(
            registry=stack.lifecycle.registry_read(),
            tool_logical_id=ME14_TOOL_LOGICAL_ID,
            tmp_path=tmp_path / "exec",
        ),
    )
    assert tool_id == ME14_TOOL_LOGICAL_ID
    assert output == ME14_OUTPUT_V1


def test_me14_c2_execution_wiring_preserves_activation_metadata(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build()
    stack.run_marketplace_tool_e2e(execution_tmp_path=tmp_path / "handoff")
    lifecycle_meta = stack.lifecycle.activation_metadata(ME14_TOOL_LOGICAL_ID)
    assert lifecycle_meta is not None
    manifest = me14_tool_proof_manifest(ME14_TOOL_LOGICAL_ID)
    environment = me14_tool_proof_environment(ME14_TOOL_LOGICAL_ID)
    host_runtime = build_harness_host_runtime(
        manifest,
        environment,
        tenant_id="tenant-me14",
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "runtime_events.db",
        application_tool_registry=stack.lifecycle.registry_read(),
        llm_adapter=FakeLLMAdapter(),
    )
    wired_meta = host_runtime.env_wiring.tool_wiring.registry.activation_metadata(
        ME14_TOOL_LOGICAL_ID,
    )
    assert wired_meta is not None
    assert wired_meta.catalog_source_id == lifecycle_meta.catalog_source_id
    assert wired_meta.package_reference == lifecycle_meta.package_reference
    assert wired_meta.version_label == lifecycle_meta.version_label
    assert wired_meta.content_digest == lifecycle_meta.content_digest


def test_me14_c2_exact_v1_release_after_wiring_closure(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build(
        listing_records=(me14_default_listing_v1(),),
    )
    evidence = stack.run_marketplace_tool_e2e(
        handoff_id="c2-handoff-v1",
        execution_tmp_path=tmp_path,
    )
    assert evidence.execution_result == ME14_OUTPUT_V1
    assert evidence.activated_version_label == ME14_VERSION_V1


def test_me14_c2_exact_v2_release_after_wiring_closure(tmp_path: Path) -> None:
    stack = MarketplaceToolExecutionProofStack.build(listing_records=(me14_listing_v2(),))
    evidence = stack.run_marketplace_tool_e2e(
        handoff_id="c2-handoff-v2",
        execution_tmp_path=tmp_path,
    )
    assert evidence.execution_result == ME14_OUTPUT_V2
    assert evidence.activated_version_label == ME14_VERSION_V2


def test_me14_c2_version_mismatch_fails_closed() -> None:
    lifecycle = ToolHostLifecycleService(host_profile_id="host-profile-me14")
    provider = Me14ToolCatalogProvider()
    service = DynamicToolAcquisitionService(
        catalog_registry=ToolCatalogProviderRegistry({provider.catalog_source_id: provider}),
        activation=lifecycle,
        materializer=Me14ToolHostActivationMaterializer(
            lifecycle.registry,
            catalog_source_id=provider.catalog_source_id,
        ),
    )
    bad = ToolDiscoveryCandidateIdentity(
        catalog_source_id=ME14_CATALOG_SOURCE_ID,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference=ME14_PACKAGE_REFERENCE_V1,
            package_version=ME14_VERSION_V2,
            package_digest=ME14_DIGEST_V1,
        ),
    )
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=CapabilitySourceIdentity(
                source_id=ME14_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id=ME14_TOOL_LOGICAL_ID,
            ),
        ),
    )
    with pytest.raises(DynamicToolAcquisitionResolutionError):
        service.acquire(
            DynamicToolAcquisitionRequest(
                operation_id="op-version-mismatch",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me14_c2_ast_gate_no_manual_invoker_in_primary_modules() -> None:
    combined = _read_primary_sources()
    for token in _FORBIDDEN_MANUAL_WIRING:
        assert token not in combined


def test_me14_c2_proof_agent_source_has_no_execution_registry_classvar() -> None:
    source = Path(
        importlib.import_module("testing_support.me14_tool_harness_execution").__file__,
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "_execution_registry":
                    raise AssertionError("proof agent must not define _execution_registry")
    assert Me14ToolProofAgent.__name__ == "Me14ToolProofAgent"
