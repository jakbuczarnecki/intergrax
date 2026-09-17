# © Artur Czarnecki. All rights reserved.

"""ME-16-C1 canonical skill wiring seam and mixed closure matrix."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from intergrax.contracts.acp_metadata_keys import AcpMetadataKey
from intergrax.contracts.capability_catalog import CapabilityKind
from intergrax.contracts.marketplace.visibility import (
    MarketplaceVisibility,
    MarketplaceVisibilityScope,
)
from intergrax.skills.errors import DynamicSkillAcquisitionResolutionError
from intergrax.tools.errors import DynamicToolAcquisitionResolutionError
from testing_support.canonical_me14_echo_tool import (
    ME14_DIGEST_V1,
    ME14_OUTPUT_V1,
    ME14_TOOL_LOGICAL_ID,
    ME14_VERSION_V2,
)
from testing_support.canonical_me15_reference_skill import (
    ME15_DIGEST_V1,
    ME15_DIGEST_V2,
    ME15_SKILL_LOGICAL_ID,
    ME15_VERSION_V2,
)
from testing_support.canonical_me16_mixed_agent import ME16_MIXED_TENANT
from testing_support.marketplace_mixed_capability_execution_composition import (
    MarketplaceMixedCapabilityProofStack,
    MixedCapabilityCompositionNotReadyError,
    me16_agent_listing_v1,
)
from testing_support.me14_tool_catalog_provider import _CustomMe14ToolCatalogProvider
from testing_support.me16_mixed_harness_execution import (
    me16_mixed_execution_environment,
    me16_mixed_execution_manifest,
    run_me16_mixed_host_execution,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_ME16_PRIMARY = (
    Path("testing_support/me16_mixed_harness_execution.py"),
    Path("testing_support/marketplace_mixed_capability_execution_composition.py"),
)
_FORBIDDEN_PRIVATE = (
    "_internal_composition",
    "_orchestration_backend",
    "_declarative_tool_invoker",
)
_FORBIDDEN_SKILL_WIRING = (
    "attach_skill_host_wiring_metadata",
    "inject_acp_skill_host_wiring_metadata",
    "HostSkillCatalogWiring(",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _primary_source() -> str:
    root = _repo_root()
    return "\n".join((root / rel).read_text(encoding="utf-8") for rel in _ME16_PRIMARY)


@pytest.fixture(autouse=True)
def _stub_host_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    from testing_support.builder import MeteringFakeLLMAdapter
    from testing_support.host_fixture_wiring import install_diagnostic_cursor_secret

    install_diagnostic_cursor_secret(monkeypatch)
    adapter = MeteringFakeLLMAdapter()

    def _resolve(
        env: object,
        agent_override: object | None = None,
        **_: object,
    ) -> object:
        del env
        if agent_override is not None:
            return agent_override
        return adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )


def test_me16_c1_task_has_no_manual_skill_wiring_metadata(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    stack.run_all_handoffs(
        discovery_correlation_id="c1-task-clean",
        agent_handoff_id="h-a",
        tool_handoff_id="h-t",
        skill_handoff_id="h-s",
    )
    config = stack.lifecycle_config
    manifest = me16_mixed_execution_manifest(
        logical_agent_id=config.logical_agent_id,
        tool_logical_id=ME14_TOOL_LOGICAL_ID,
        application_id=config.application_id,
    )
    env = me16_mixed_execution_environment(
        tool_logical_id=ME14_TOOL_LOGICAL_ID,
        skill_profile=stack.skill_lifecycle.skill_profile,
        environment_id=config.environment_id,
    )
    from intergrax.applications._shared.harness_host_runtime import build_harness_host_runtime
    from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
    from intergrax.runtime.task.task import Task, TaskContext
    from testing_support.agent_platform_admin_harness import lifecycle_proof_durable_profile_stores
    from testing_support.builder import FakeLLMAdapter
    from testing_support.canonical_me16_mixed_agent import (
        ME16_MIXED_CAPABILITY,
        ME16_MIXED_CONTRACT_ID,
        ME16_MIXED_TASK_INPUT,
    )

    profile_stores = lifecycle_proof_durable_profile_stores(stack.agent_stack.runtime_root)
    host = build_harness_host_runtime(
        manifest,
        env,
        tenant_id=ME16_MIXED_TENANT,
        registry_projection=stack.agent_stack.resolve_serving_projection(),
        trace_db_path=tmp_path / "trace.db",
        runtime_events_db_path=tmp_path / "events.db",
        document_store=InMemoryDocumentStore(),
        revision_store=profile_stores.revision_store,
        pinning_store=profile_stores.pinning_store,
        active_store=profile_stores.active_store,
        application_tool_registry=stack.tool_lifecycle.registry_read(),
        application_skill_registry=stack.skill_lifecycle.registry,
        llm_adapter=FakeLLMAdapter(),
    )
    task = Task(
        tenant_id=ME16_MIXED_TENANT,
        user_id="u",
        message=ME16_MIXED_TASK_INPUT,
        agent_id=ME16_MIXED_CONTRACT_ID,
        context=TaskContext(capability=ME16_MIXED_CAPABILITY),
    )
    assert AcpMetadataKey.SKILL_HOST_WIRING not in task.metadata
    asyncio.run(host.execution.execute(task))


def test_me16_c1_runtime_owns_skill_host_wiring_injection(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    stack.run_all_handoffs(
        discovery_correlation_id="c1-inject",
        agent_handoff_id="h-a2",
        tool_handoff_id="h-t2",
        skill_handoff_id="h-s2",
    )
    result = asyncio.run(
        run_me16_mixed_host_execution(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "exec",
        ),
    )
    assert result.answer is not None
    assert ME14_OUTPUT_V1.split("|")[0] in result.answer or "|" in result.answer


def test_me16_c1_application_skill_registry_reaches_agent_runtime_context(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    stack.run_all_handoffs(
        discovery_correlation_id="c1-ctx",
        agent_handoff_id="h-a3",
        tool_handoff_id="h-t3",
        skill_handoff_id="h-s3",
    )
    _, answer, _ = asyncio.run(
        __import__(
            "testing_support.me16_mixed_harness_execution",
            fromlist=["execute_me16_mixed_via_host_execution_engine"],
        ).execute_me16_mixed_via_host_execution_engine(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "e2e",
        ),
    )
    assert answer


def test_me16_c1_skill_binding_metadata_survives_host_composition(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    evidence = stack.run_marketplace_mixed_e2e(execution_tmp_path=tmp_path / "bind")
    binding = stack.skill_lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
    assert binding is not None
    assert binding.version_label == evidence.skill_bound_version


@pytest.mark.parametrize(
    ("handoffs", "expect_agent", "expect_tool", "expect_skill"),
    [
        (("agent",), True, False, False),
        (("agent", "tool"), True, True, False),
        (("agent", "skill"), True, False, True),
        (("tool", "skill"), False, True, True),
    ],
    ids=["agent_only", "agent_tool", "agent_skill", "tool_skill"],
)
def test_me16_c1_partial_readiness_blocks_execution(
    tmp_path: Path,
    handoffs: tuple[str, ...],
    expect_agent: bool,
    expect_tool: bool,
    expect_skill: bool,
) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    corr = "c1-partial"
    if "agent" in handoffs:
        stack.handoff_agent(
            discovery_correlation_id=corr,
            selection_id="s-a",
            handoff_id="h-pa",
        )
    if "tool" in handoffs:
        stack.handoff_tool(
            discovery_correlation_id=corr,
            selection_id="s-t",
            handoff_id="h-pt",
        )
    if "skill" in handoffs:
        stack.handoff_skill(
            discovery_correlation_id=corr,
            selection_id="s-s",
            handoff_id="h-ps",
        )
    readiness = stack.readiness()
    assert readiness.agent_ready is expect_agent
    assert readiness.tool_ready is expect_tool
    assert readiness.skill_ready is expect_skill
    assert not readiness.execution_allowed
    with pytest.raises(MixedCapabilityCompositionNotReadyError):
        stack.assert_execution_readiness()


def test_me16_c1_all_three_capabilities_allow_execution(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    with pytest.raises(MixedCapabilityCompositionNotReadyError):
        stack.assert_execution_readiness()
    stack.run_all_handoffs(
        discovery_correlation_id="c1-full",
        agent_handoff_id="h-fa",
        tool_handoff_id="h-ft",
        skill_handoff_id="h-fs",
    )
    assert stack.assert_execution_readiness().execution_allowed
    asyncio.run(
        __import__(
            "testing_support.me16_mixed_harness_execution",
            fromlist=["execute_me16_mixed_via_host_execution_engine"],
        ).execute_me16_mixed_via_host_execution_engine(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "full",
        ),
    )


def test_me16_c1_org_private_mixed_isolation(tmp_path: Path) -> None:
    config = MarketplaceMixedCapabilityProofStack.build(tmp_path).lifecycle_config
    org = "org-me16-private"
    from testing_support.marketplace_tool_execution_composition import _marketplace_listing_record as tool_rec
    from testing_support.marketplace_skill_composition import _marketplace_listing_record as skill_rec
    from intergrax.marketplace.handoff_traceability.errors import MarketplaceHandoffSelectionError

    from testing_support.canonical_me14_echo_tool import ME14_DIGEST_V1, ME14_VERSION_V1
    from testing_support.canonical_me15_reference_skill import ME15_DIGEST_V1, ME15_VERSION_V1

    from dataclasses import replace

    agent_listing = replace(
        me16_agent_listing_v1(config),
        visibility=MarketplaceVisibility(
            scope=MarketplaceVisibilityScope.ORGANIZATION_PRIVATE,
            organization_id=org,
        ),
    )
    stack = MarketplaceMixedCapabilityProofStack.build(
        tmp_path,
        agent_listing_records=(agent_listing,),
        tool_listing_records=(
            tool_rec(
                version_label=ME14_VERSION_V1,
                content_digest=ME14_DIGEST_V1,
                organization_id=org,
            ),
        ),
        skill_listing_records=(
            skill_rec(
                version_label=ME15_VERSION_V1,
                content_digest=ME15_DIGEST_V1,
                organization_id=org,
            ),
        ),
    )
    with pytest.raises(MarketplaceHandoffSelectionError):
        stack.handoff_agent(
            discovery_correlation_id="org-iso",
            selection_id="s-org",
            handoff_id="h-org-a",
            tenant_id="foreign-tenant",
        )


def test_me16_c1_tool_version_mismatch_blocks_execution(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    lifecycle = stack.tool_lifecycle
    from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
    from testing_support.me14_tool_catalog_provider import ME14_CATALOG_SOURCE_ID
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest

    service = stack.tool_acquisition
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
    bad = ToolDiscoveryCandidateIdentity(
        catalog_source_id=ME14_CATALOG_SOURCE_ID,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference="pkg",
            package_version=ME14_VERSION_V2,
            package_digest=ME14_DIGEST_V1,
        ),
    )
    with pytest.raises(DynamicToolAcquisitionResolutionError):
        service.acquire(
            DynamicToolAcquisitionRequest(
                operation_id="me16-c1-tool-version",
                host_profile_id=lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me16_c1_tool_digest_mismatch_blocks_execution(tmp_path: Path) -> None:
    from intergrax.tools.dynamic_acquisition import DynamicToolAcquisitionRequest
    from intergrax.tools.identity import ToolDiscoveryCandidateIdentity, ToolPackageCandidate
    from testing_support.me14_tool_catalog_provider import ME14_CATALOG_SOURCE_ID
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
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
    bad = ToolDiscoveryCandidateIdentity(
        catalog_source_id=ME14_CATALOG_SOURCE_ID,
        package=ToolPackageCandidate(
            logical_tool_id=ME14_TOOL_LOGICAL_ID,
            package_reference="pkg",
            package_version="1.0.0",
            package_digest=ME14_DIGEST_V1 + "-bad",
        ),
    )
    with pytest.raises(DynamicToolAcquisitionResolutionError):
        stack.tool_acquisition.acquire(
            DynamicToolAcquisitionRequest(
                operation_id="me16-c1-tool-digest",
                host_profile_id=stack.tool_lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me16_c1_skill_version_mismatch_blocks_execution(tmp_path: Path) -> None:
    from intergrax.skills.dynamic_acquisition import DynamicSkillAcquisitionRequest
    from intergrax.skills.identity import SkillDiscoveryCandidateIdentity, SkillPackageCandidate
    from testing_support.me15_skill_catalog_provider import ME15_CATALOG_SOURCE_ID
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=CapabilitySourceIdentity(
                source_id=ME15_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=ME15_SKILL_LOGICAL_ID,
            ),
        ),
    )
    bad = SkillDiscoveryCandidateIdentity(
        catalog_source_id=ME15_CATALOG_SOURCE_ID,
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference="pkg",
            package_version=ME15_VERSION_V2,
            package_digest=ME15_DIGEST_V1,
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        stack.skill_acquisition.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="me16-c1-skill-version",
                host_profile_id=stack.skill_lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me16_c1_skill_digest_mismatch_blocks_execution(tmp_path: Path) -> None:
    from intergrax.skills.dynamic_acquisition import DynamicSkillAcquisitionRequest
    from intergrax.skills.identity import SkillDiscoveryCandidateIdentity, SkillPackageCandidate
    from testing_support.me15_skill_catalog_provider import ME15_CATALOG_SOURCE_ID
    from intergrax.contracts.capability_catalog import (
        CapabilityDiscoveryIdentity,
        CapabilityKind,
        CapabilityLogicalIdentity,
        CapabilitySourceIdentity,
        CapabilitySourceKind,
    )
    from intergrax.contracts.capability_catalog.identity_key import CapabilityIdentityKey

    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    identity_key = CapabilityIdentityKey.from_discovery_identity(
        CapabilityDiscoveryIdentity(
            kind=CapabilityKind.SKILL,
            source=CapabilitySourceIdentity(
                source_id=ME15_CATALOG_SOURCE_ID,
                source_kind=CapabilitySourceKind.OFFICIAL,
            ),
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.SKILL,
                logical_id=ME15_SKILL_LOGICAL_ID,
            ),
        ),
    )
    bad = SkillDiscoveryCandidateIdentity(
        catalog_source_id=ME15_CATALOG_SOURCE_ID,
        package=SkillPackageCandidate(
            logical_skill_id=ME15_SKILL_LOGICAL_ID,
            package_reference="pkg",
            package_version="1.0.0",
            package_digest=ME15_DIGEST_V2,
        ),
    )
    with pytest.raises(DynamicSkillAcquisitionResolutionError):
        stack.skill_acquisition.acquire(
            DynamicSkillAcquisitionRequest(
                operation_id="me16-c1-skill-digest",
                host_profile_id=stack.skill_lifecycle.host_profile_id,
                capability_identity_key=identity_key,
                selected_identity=bad,
            ),
        )


def test_me16_c1_duplicate_mixed_handoffs_do_not_repeat_lifecycle_effects(tmp_path: Path) -> None:
    stack = MarketplaceMixedCapabilityProofStack.build(tmp_path)
    kwargs = dict(
        discovery_correlation_id="dup",
        agent_selection_id="sel-dup-a",
        tool_selection_id="sel-dup-t",
        skill_selection_id="sel-dup-s",
        agent_handoff_id="handoff-dup-a",
        tool_handoff_id="handoff-dup-t",
        skill_handoff_id="handoff-dup-s",
    )
    stack.run_all_handoffs(**kwargs)
    tool_version = stack.tool_lifecycle.activation_metadata(ME14_TOOL_LOGICAL_ID)
    skill_version = stack.skill_lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID)
    pointer_before = stack.agent_stack.admin.inspect_serving(
        application_id=stack.lifecycle_config.application_id,
        application_environment_id=stack.lifecycle_config.environment_id,
    ).serving_pointer_revision
    from intergrax.contracts.marketplace.handoff_traceability import (
        CapabilityHandoffIdentityConflictError,
    )

    with pytest.raises(CapabilityHandoffIdentityConflictError):
        stack.run_all_handoffs(**kwargs)
    assert (
        stack.tool_lifecycle.activation_metadata(ME14_TOOL_LOGICAL_ID) == tool_version
    )
    assert (
        stack.skill_lifecycle.binding_metadata(ME15_SKILL_LOGICAL_ID) == skill_version
    )
    pointer_after = stack.agent_stack.admin.inspect_serving(
        application_id=stack.lifecycle_config.application_id,
        application_environment_id=stack.lifecycle_config.environment_id,
    ).serving_pointer_revision
    assert pointer_after == pointer_before


def test_me16_c1_custom_tool_provider_reaches_execution(tmp_path: Path) -> None:
    custom = _CustomMe14ToolCatalogProvider()
    stack = MarketplaceMixedCapabilityProofStack.build(
        tmp_path,
        tool_catalog_provider=custom,
    )
    stack.run_all_handoffs(
        discovery_correlation_id="custom-prov",
        agent_handoff_id="h-ca",
        tool_handoff_id="h-ct",
        skill_handoff_id="h-cs",
    )
    _, answer, _ = asyncio.run(
        __import__(
            "testing_support.me16_mixed_harness_execution",
            fromlist=["execute_me16_mixed_via_host_execution_engine"],
        ).execute_me16_mixed_via_host_execution_engine(
            agent_stack=stack.agent_stack,
            tool_registry=stack.tool_lifecycle.registry_read(),
            skill_lifecycle=stack.skill_lifecycle,
            tmp_path=tmp_path / "custom",
        ),
    )
    assert ME14_OUTPUT_V1 in answer or "|" in answer


def test_me16_c1_primary_path_has_no_private_execution_access() -> None:
    combined = _primary_source()
    for token in _FORBIDDEN_PRIVATE:
        assert token not in combined


def test_me16_c1_primary_path_has_no_manual_skill_wiring_injection() -> None:
    combined = _primary_source()
    for token in _FORBIDDEN_SKILL_WIRING:
        assert token not in combined


def test_me16_c1_no_nexus_import_in_primary_modules() -> None:
    for rel in _ME16_PRIMARY:
        tree = ast.parse((_repo_root() / rel).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("intergrax.runtime.nexus")


def test_me16_c1_agent_only_is_not_ready(tmp_path: Path) -> None:
    test_me16_c1_partial_readiness_blocks_execution(
        tmp_path,
        ("agent",),
        True,
        False,
        False,
    )


def test_me16_c1_agent_tool_without_skill_is_not_ready(tmp_path: Path) -> None:
    test_me16_c1_partial_readiness_blocks_execution(
        tmp_path,
        ("agent", "tool"),
        True,
        True,
        False,
    )


def test_me16_c1_agent_skill_without_tool_is_not_ready(tmp_path: Path) -> None:
    test_me16_c1_partial_readiness_blocks_execution(
        tmp_path,
        ("agent", "skill"),
        True,
        False,
        True,
    )


def test_me16_c1_tool_skill_without_agent_is_not_ready(tmp_path: Path) -> None:
    test_me16_c1_partial_readiness_blocks_execution(
        tmp_path,
        ("tool", "skill"),
        False,
        True,
        True,
    )
