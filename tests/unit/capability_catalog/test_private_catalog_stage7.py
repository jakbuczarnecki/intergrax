# © Artur Czarnecki. All rights reserved.

"""Stage 7 private catalog federation, discovery, and air-gapped proof tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from intergrax.capability_catalog import (
    CapabilityCatalogConfigurationError,
    CapabilityCatalogIdentityConflict,
    CapabilityCatalogSourceFailure,
    FederatedCapabilityCatalog,
    discover_capability_candidates,
    merge_capability_catalog_entries,
)
from intergrax.capability_catalog.adapters.private_skill import (
    PrivateSkillCapabilityCatalogSource,
    PrivateSkillCatalogPackage,
)
from intergrax.capability_catalog.adapters.private_tool import (
    PrivateToolCapabilityCatalogSource,
    PrivateToolCatalogRecord,
)
from intergrax.capability_catalog.adapters.skill import SkillBundleCatalogSource
from intergrax.capability_catalog.adapters.tool import ToolBundleCatalogSource
from intergrax.capability_catalog.entry import CapabilityCatalogEntry
from intergrax.capability_catalog.source import CapabilityCatalogSource
from intergrax.contracts.capability_catalog import (
    AvailabilityDisposition,
    CapabilityDiscoveryIdentity,
    CapabilityDiscoveryQuery,
    CapabilityDiscoveryScope,
    CapabilityDiscoveryScopeMode,
    CapabilityKind,
    CapabilityLogicalIdentity,
    CapabilityProvenance,
    CapabilitySourceIdentity,
    CapabilitySourceKind,
    LogicalIdentityFilter,
)
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.catalog import ToolBundleEntry, clear_tool_catalog, register_tool_bundle
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class _In(BaseModel):
    x: int


class _Out(BaseModel):
    y: int


@pytest.fixture(autouse=True)
def _isolated_tool_catalog() -> None:
    clear_tool_catalog()
    yield
    clear_tool_catalog()


def _discovery_scope() -> CapabilityDiscoveryScope:
    return CapabilityDiscoveryScope(mode=CapabilityDiscoveryScopeMode.GLOBAL)


def _discovery_query(**kwargs: object) -> CapabilityDiscoveryQuery:
    return CapabilityDiscoveryQuery(scope=_discovery_scope(), **kwargs)


def _enterprise_tool_source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="enterprise.acme.tools",
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
    )


def _enterprise_skill_source() -> CapabilitySourceIdentity:
    return CapabilitySourceIdentity(
        source_id="enterprise.acme.skills",
        source_kind=CapabilitySourceKind.ENTERPRISE_PRIVATE,
    )


def _private_tool_source() -> PrivateToolCapabilityCatalogSource:
    return PrivateToolCapabilityCatalogSource(
        source=_enterprise_tool_source(),
        records=(
            PrivateToolCatalogRecord(
                logical_id="tool.enterprise.search",
                version_label="3.2.1",
                package_reference="private://tools/search/3.2.1",
            ),
            PrivateToolCatalogRecord(
                logical_id="foo.search",
                version_label="1.0.0",
            ),
        ),
    )


def _private_skill_source() -> PrivateSkillCapabilityCatalogSource:
    return PrivateSkillCapabilityCatalogSource(
        source=_enterprise_skill_source(),
        packages=(
            PrivateSkillCatalogPackage(
                manifest=SkillManifest(
                    skill_id="skill.enterprise.research",
                    version="2.4.0",
                    description="Enterprise research skill",
                ),
                package_reference="private://skills/research/2.4.0",
            ),
            PrivateSkillCatalogPackage(
                manifest=SkillManifest(
                    skill_id="foo.search",
                    version="2.1.0",
                    description="Shared logical id skill",
                ),
            ),
        ),
    )


def _register_builtin_tool_foo_search() -> None:
    def _register(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
        del ctx
        handler = type("H", (), {"execute": lambda self, req: _Out(y=req.input.x)})()
        registry.register(
            ToolContract(
                tool_id="foo.search",
                name="foo.search",
                description="builtin foo search",
                input_schema=_In,
                output_schema=_Out,
                error_mapping={},
                side_effects=False,
            ),
            handler,
        )

    register_tool_bundle(
        ToolBundleEntry(
            bundle_id="foo",
            tool_ids=("foo.search",),
            register=_register,
            description="Foo tools",
        ),
    )


class _FailingPrivateSource:
    @property
    def source_id(self) -> str:
        return "enterprise.fail.tools"

    def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
        raise RuntimeError("private catalog backend unavailable")


def test_air_gapped_private_tool_and_skill_federation_and_discovery() -> None:
    _register_builtin_tool_foo_search()
    federated = FederatedCapabilityCatalog(
        (
            ToolBundleCatalogSource(),
            _private_tool_source(),
            SkillBundleCatalogSource(),
            _private_skill_source(),
        ),
    )
    snapshot = federated.snapshot()
    logical_ids = {entry.identity.logical.logical_id for entry in snapshot.entries}
    assert "tool.enterprise.search" in logical_ids
    assert "skill.enterprise.research" in logical_ids
    assert "foo.search" in logical_ids

    tool_candidates = discover_capability_candidates(
        snapshot,
        _discovery_query(
            kinds=(CapabilityKind.TOOL,),
            logical_identity=LogicalIdentityFilter(
                exact_logical_ids=("tool.enterprise.search",),
            ),
        ),
    )
    assert len(tool_candidates) == 1
    assert tool_candidates[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE
    assert tool_candidates[0].catalog_entry.provenance.version_label == "3.2.1"

    skill_candidates = discover_capability_candidates(
        snapshot,
        _discovery_query(
            kinds=(CapabilityKind.SKILL,),
            logical_identity=LogicalIdentityFilter(
                exact_logical_ids=("skill.enterprise.research",),
            ),
        ),
    )
    assert len(skill_candidates) == 1
    assert skill_candidates[0].catalog_entry.provenance.version_label == "2.4.0"
    assert skill_candidates[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE


def test_same_logical_id_across_builtin_and_private_sources_is_not_conflict() -> None:
    _register_builtin_tool_foo_search()
    federated = FederatedCapabilityCatalog(
        (
            ToolBundleCatalogSource(),
            _private_tool_source(),
        ),
    )
    snapshot = federated.snapshot()
    foo_entries = [
        entry
        for entry in snapshot.entries
        if entry.identity.logical.logical_id == "foo.search"
    ]
    assert len(foo_entries) == 2
    source_ids = {entry.identity.source.source_id for entry in foo_entries}
    assert source_ids == {"enterprise.acme.tools", "tools.catalog.builtin"}


def test_same_source_conflicting_entries_fail_closed_at_federation() -> None:
    source = _enterprise_tool_source()
    first = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id="tool.dup"),
        ),
        provenance=CapabilityProvenance(source=source, version_label="1.0.0"),
    )
    second = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(kind=CapabilityKind.TOOL, logical_id="tool.dup"),
        ),
        provenance=CapabilityProvenance(source=source, version_label="2.0.0"),
    )

    class _DuplicateConflictSource:
        @property
        def source_id(self) -> str:
            return source.source_id

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            return (first, second)

    with pytest.raises(CapabilityCatalogIdentityConflict):
        FederatedCapabilityCatalog((_DuplicateConflictSource(),)).snapshot()


def test_private_source_failure_aborts_federation() -> None:
    with pytest.raises(CapabilityCatalogSourceFailure, match="enterprise.fail.tools"):
        FederatedCapabilityCatalog(
            (
                _private_tool_source(),
                _FailingPrivateSource(),
            ),
        ).snapshot()


def test_discovery_does_not_mutate_tool_or_skill_registries() -> None:
    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()
    tool_before = tuple(tool_registry.tool_ids())
    skill_before = tuple(skill_registry.skill_ids())
    tool_profile = ToolProfile(enabled=["existing.tool"])
    skill_profile = SkillProfile(enabled=["existing.skill"])
    tool_profile_before = tool_profile.model_dump()
    skill_profile_before = skill_profile.model_dump()

    federated = FederatedCapabilityCatalog(
        (
            _private_tool_source(),
            _private_skill_source(),
        ),
    )
    snapshot = federated.snapshot()
    discover_capability_candidates(snapshot, _discovery_query())

    assert tuple(tool_registry.tool_ids()) == tool_before
    assert tuple(skill_registry.skill_ids()) == skill_before
    assert tool_profile.model_dump() == tool_profile_before
    assert skill_profile.model_dump() == skill_profile_before


def test_private_tool_exists_in_discovery_but_not_in_runtime_registry() -> None:
    tool_registry = ToolRegistry()
    tool_profile = ToolProfile()

    snapshot = FederatedCapabilityCatalog((_private_tool_source(),)).snapshot()
    candidates = discover_capability_candidates(
        snapshot,
        _discovery_query(
            kinds=(CapabilityKind.TOOL,),
            logical_identity=LogicalIdentityFilter(
                exact_logical_ids=("tool.enterprise.search",),
            ),
        ),
    )
    assert len(candidates) == 1
    assert candidates[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE
    assert "tool.enterprise.search" not in tool_registry.tool_ids()
    assert "tool.enterprise.search" not in tool_profile.enabled


def test_private_skill_exists_in_discovery_but_not_in_runtime_registry() -> None:
    skill_registry = SkillRegistry()
    skill_profile = SkillProfile()

    snapshot = FederatedCapabilityCatalog((_private_skill_source(),)).snapshot()
    candidates = discover_capability_candidates(
        snapshot,
        _discovery_query(
            kinds=(CapabilityKind.SKILL,),
            logical_identity=LogicalIdentityFilter(
                exact_logical_ids=("skill.enterprise.research",),
            ),
        ),
    )
    assert len(candidates) == 1
    assert candidates[0].catalog_entry.provenance.version_label == "2.4.0"
    assert candidates[0].availability is AvailabilityDisposition.CATALOG_AVAILABLE
    assert "skill.enterprise.research" not in skill_registry.skill_ids()
    assert "skill.enterprise.research" not in skill_profile.enabled


def test_private_source_constructor_rejects_multi_version_same_logical_id() -> None:
    with pytest.raises(CapabilityCatalogConfigurationError, match="conflicting version metadata"):
        PrivateToolCapabilityCatalogSource(
            source=_enterprise_tool_source(),
            records=(
                PrivateToolCatalogRecord(
                    logical_id="tool.enterprise.search",
                    version_label="1.0.0",
                ),
                PrivateToolCatalogRecord(
                    logical_id="tool.enterprise.search",
                    version_label="2.0.0",
                ),
            ),
        )


def test_federated_snapshot_preserves_deterministic_order() -> None:
    _register_builtin_tool_foo_search()
    first = FederatedCapabilityCatalog(
        (
            ToolBundleCatalogSource(),
            _private_tool_source(),
            SkillBundleCatalogSource(),
            _private_skill_source(),
        ),
    ).snapshot()
    second = FederatedCapabilityCatalog(
        (
            SkillBundleCatalogSource(),
            _private_skill_source(),
            ToolBundleCatalogSource(),
            _private_tool_source(),
        ),
    ).snapshot()
    assert first.entries == second.entries


def test_merge_preserves_source_qualified_identities_from_private_sources() -> None:
    source = _enterprise_tool_source()
    entry = CapabilityCatalogEntry(
        identity=CapabilityDiscoveryIdentity(
            kind=CapabilityKind.TOOL,
            source=source,
            logical=CapabilityLogicalIdentity(
                kind=CapabilityKind.TOOL,
                logical_id="tool.enterprise.search",
            ),
        ),
        provenance=CapabilityProvenance(
            source=source,
            version_label="3.2.1",
            package_reference="private://tools/search/3.2.1",
        ),
    )

    class _StaticPrivateSource(CapabilityCatalogSource):
        @property
        def source_id(self) -> str:
            return source.source_id

        def read_entries(self) -> tuple[CapabilityCatalogEntry, ...]:
            return (entry,)

    merged = merge_capability_catalog_entries(
        ((source.source_id, entry), (source.source_id, entry)),
    )
    assert len(merged) == 1
    assert merged[0].provenance.package_reference == "private://tools/search/3.2.1"
