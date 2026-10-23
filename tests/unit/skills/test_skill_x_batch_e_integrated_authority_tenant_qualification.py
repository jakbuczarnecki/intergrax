# © Artur Czarnecki. All rights reserved.

"""SKILL-X Batch E — integrated authority, cross-tenant durable adversarial, replay trust."""

from __future__ import annotations

import dataclasses
import json
import threading

import pytest

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.nexus.tools.tool_selection import (
    SkillPackSelectionStrategy,
    ToolSelectionContext,
)
from intergrax.skills.contracts.skill_runtime_binding_metadata import SkillRuntimeBindingMetadata
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.execution_binding import (
    SkillExecutionPinningConflictError,
    SkillExecutionPinningStorageError,
    bind_resolved_skill_pack,
    recover_bound_skill_pack,
    resolve_bound_skill_pack,
)
from intergrax.skills.execution_pinning_persistence import (
    DocumentStoreSkillExecutionPinningStore,
    KvSkillExecutionPinningStore,
    decode_skill_execution_binding,
    encode_skill_execution_binding,
)
from intergrax.skills.host_lifecycle import SkillHostLifecycleService
from intergrax.skills.integration.contract_resolution import (
    resolve_contract_tools,
    resolve_gateway_allowed_tools,
)
from intergrax.skills.registry.profile import SkillProfile
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.skills.resolver import SkillResolutionError, SkillResolver
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry
from testing_support.builder import tools_agent_make_contract
from tests.unit.skills.test_skill_x_batch_c_versioning_provenance_recovery import (
    InMemoryConditionalDocumentStore,
    InMemoryKVStore,
    _binding_metadata,
    _manifest,
)

pytestmark = pytest.mark.unit

_SKILL_A = SkillManifest(skill_id="a.pack", description="a", tool_ids=("tool.a",))
_SKILL_B = SkillManifest(skill_id="b.pack", description="b", tool_ids=("tool.b",))


def _contract_with_skills(*skills: SkillManifest) -> AgentContract:
    return AgentContract(
        id="batch-e-agent",
        name="Batch E",
        description="batch e",
        capabilities=["batch.e"],
        skills=list(skills),
    )


def _pinning_backends():
    return (
        ("kv", KvSkillExecutionPinningStore(InMemoryKVStore())),
        (
            "document",
            DocumentStoreSkillExecutionPinningStore(InMemoryConditionalDocumentStore()),
        ),
    )


# --- EC2: Skill → proposal → Tool selection → Governance ceiling (negative) ---


def test_integrated_unauthorized_tool_proposal_clamped_by_plan_ceiling() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("evil.pack", tool_ids=("tool.evil", "tool.allowed")))
    tools = ToolRegistry()
    tools.register(
        tools_agent_make_contract("tool.allowed", type("In", (), {}), type("Out", (), {})),
        object(),
    )
    tools.register(
        tools_agent_make_contract("tool.evil", type("In", (), {}), type("Out", (), {})),
        object(),
    )
    execution_id = mint_execution_id()
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        strategy = SkillPackSelectionStrategy()
        selected = strategy.select_tool_ids(
            ToolSelectionContext(
                registry=tools,
                query="",
                skill_profile=SkillProfile(enabled=["evil.pack"]),
                skill_registry=registry,
                skill_pinning_store=pinning,
                tenant_id="tenant-a",
                plan_allowed_tool_ids=("tool.allowed",),
            ),
        )
        assert selected == ("tool.allowed",)
        assert "tool.evil" not in selected
    finally:
        reset_active_execution_identity(token)


def test_integrated_skill_tool_outside_host_tool_profile_fail_closed() -> None:
    skills = SkillRegistry()
    skills.register(_SKILL_A)
    contract = _contract_with_skills(_SKILL_A)
    with pytest.raises(SkillResolutionError, match="exceed host ToolProfile"):
        resolve_contract_tools(
            contract,
            skill_resolver=SkillResolver(skills),
            skill_profile=SkillProfile(enabled=["a.pack"]),
            tool_profile=ToolProfile(enabled=["tool.other"]),
        )


def test_integrated_forged_metadata_cannot_authorize_extra_tools() -> None:
    assert resolve_gateway_allowed_tools(["tool.a"], ["tool.evil", "tool.a"]) == ["tool.a"]


def test_integrated_disabled_skill_not_effective_for_agent_contract() -> None:
    skills = SkillRegistry()
    skills.register(_SKILL_A)
    skills.register(_SKILL_B)
    contract = _contract_with_skills(_SKILL_A, _SKILL_B)
    with pytest.raises(SkillResolutionError, match="not enabled"):
        resolve_contract_tools(
            contract,
            skill_resolver=SkillResolver(skills),
            skill_profile=SkillProfile(enabled=["a.pack"]),
        )


def test_integrated_enabled_skill_not_on_agent_contract_not_applicable() -> None:
    skills = SkillRegistry()
    skills.register(_SKILL_A)
    skills.register(_SKILL_B)
    contract = _contract_with_skills(_SKILL_A)
    updated, pack = resolve_contract_tools(
        contract,
        skill_resolver=SkillResolver(skills),
        skill_profile=SkillProfile(enabled=["a.pack", "b.pack"]),
    )
    assert updated.allowed_tools == ["tool.a"]
    assert pack.skill_ids == ("a.pack",)


def test_integrated_prompt_instruction_does_not_widen_tool_surface() -> None:
    manifest = SkillManifest(
        skill_id="prompt.pack",
        description="prompt only",
        tool_ids=(),
        prompt_instruction_ids=("prompt.demand.external_side_effect",),
    )
    registry = SkillRegistry()
    registry.register(manifest)
    pack = SkillResolver(registry).resolve_composition(["prompt.pack"]).pack
    assert not pack.tool_ids
    assert "prompt.demand.external_side_effect" in pack.prompt_instruction_ids


def test_integrated_hitl_skill_tools_remain_under_plan_ceiling() -> None:
    registry = SkillRegistry()
    registry.register(
        SkillManifest(
            skill_id="hitl.pack",
            description="hitl",
            tool_ids=("hitl.list_pending", "hitl.approve"),
        ),
    )
    tools = ToolRegistry()
    for tool_id in ("hitl.list_pending", "hitl.approve"):
        tools.register(
            tools_agent_make_contract(tool_id, type("In", (), {}), type("Out", (), {})),
            object(),
        )
    execution_id = mint_execution_id()
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        strategy = SkillPackSelectionStrategy()
        selected = strategy.select_tool_ids(
            ToolSelectionContext(
                registry=tools,
                query="",
                skill_profile=SkillProfile(enabled=["hitl.pack"]),
                skill_registry=registry,
                skill_pinning_store=pinning,
                tenant_id="tenant-a",
                plan_allowed_tool_ids=("hitl.list_pending",),
            ),
        )
        assert selected == ("hitl.list_pending",)
    finally:
        reset_active_execution_identity(token)


def test_integrated_installed_not_enabled_blocks_enable_without_install() -> None:
    lifecycle = SkillHostLifecycleService(host_profile_id="batch-e-host")
    with pytest.raises(ValueError, match="not installed"):
        lifecycle.enable_installed_skill("missing.skill")


def test_integrated_catalog_substitution_blocks_canonical_tool_selection_path() -> None:
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.old",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.register_or_replace(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.evil",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:substituted"),
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="catalog witness"):
            resolve_bound_skill_pack(
                tenant_id="tenant-a",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=pinning,
            )
    finally:
        reset_active_execution_identity(token)


def test_integrated_recovery_required_without_pin_fail_closed() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    execution_id = mint_execution_id()
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="no bound skill pack"):
            SkillPackSelectionStrategy().select_tool_ids(
                ToolSelectionContext(
                    registry=ToolRegistry(),
                    query="",
                    skill_profile=SkillProfile(enabled=["a.pack"]),
                    skill_registry=registry,
                    skill_pinning_store=KvSkillExecutionPinningStore(InMemoryKVStore()),
                    tenant_id="tenant-a",
                    skill_execution_recovery_required=True,
                ),
            )
    finally:
        reset_active_execution_identity(token)


# --- EC8: cross-tenant durable adversarial (KV + document) ---


@pytest.mark.parametrize("backend_name,store_factory", _pinning_backends())
def test_ec8_same_execution_id_isolated_per_tenant(
    backend_name: str,
    store_factory: KvSkillExecutionPinningStore | DocumentStoreSkillExecutionPinningStore,
) -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", tool_ids=("tool.a",)))
    registry.register(_manifest("b.pack", tool_ids=("tool.b",)))
    comp_a = SkillResolver(registry).resolve_composition(["a.pack"])
    comp_b = SkillResolver(registry).resolve_composition(["b.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=store_factory,
        resolved_composition=comp_a,
    )
    bind_resolved_skill_pack(
        tenant_id="tenant-b",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["b.pack"]),
        skill_registry=registry,
        pinning_store=store_factory,
        resolved_composition=comp_b,
    )
    pack_a = recover_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store_factory,
    )
    pack_b = recover_bound_skill_pack(
        tenant_id="tenant-b",
        execution_id=execution_id,
        pinning_store=store_factory,
    )
    assert pack_a.skill_ids == ("a.pack",)
    assert pack_b.skill_ids == ("b.pack",)
    assert store_factory.get(tenant_id="tenant-c", execution_id=execution_id) is None


@pytest.mark.parametrize("backend_name,store_factory", _pinning_backends())
def test_ec8_cross_tenant_read_returns_none(
    backend_name: str,
    store_factory: KvSkillExecutionPinningStore | DocumentStoreSkillExecutionPinningStore,
) -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=store_factory,
        resolved_composition=composition,
    )
    assert store_factory.get(tenant_id="tenant-b", execution_id=execution_id) is None


@pytest.mark.parametrize("backend_name,store_factory", _pinning_backends())
def test_ec8_identical_retry_idempotent_per_tenant(
    backend_name: str,
    store_factory: KvSkillExecutionPinningStore | DocumentStoreSkillExecutionPinningStore,
) -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    profile = SkillProfile(enabled=["a.pack"])
    for tenant_id in ("tenant-a", "tenant-b"):
        first = bind_resolved_skill_pack(
            tenant_id=tenant_id,
            execution_id=execution_id,
            skill_profile=profile,
            skill_registry=registry,
            pinning_store=store_factory,
            resolved_composition=composition,
        )
        second = bind_resolved_skill_pack(
            tenant_id=tenant_id,
            execution_id=execution_id,
            skill_profile=profile,
            skill_registry=registry,
            pinning_store=store_factory,
            resolved_composition=composition,
        )
        assert first.resolved_pack.snapshot_digest == second.resolved_pack.snapshot_digest


@pytest.mark.parametrize("backend_name,store_factory", _pinning_backends())
def test_ec8_conflicting_rebind_rejected_within_tenant(
    backend_name: str,
    store_factory: KvSkillExecutionPinningStore | DocumentStoreSkillExecutionPinningStore,
) -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack", "1.0.0", tool_ids=("tool.one",)))
    first = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=store_factory,
        resolved_composition=first,
    )
    registry.register_or_replace(_manifest("a.pack", "2.0.0", tool_ids=("tool.two",)))
    second = SkillResolver(registry).resolve_composition(["a.pack"])
    with pytest.raises(SkillExecutionPinningConflictError, match="already pinned"):
        bind_resolved_skill_pack(
            tenant_id="tenant-a",
            execution_id=execution_id,
            skill_profile=SkillProfile(enabled=["a.pack"]),
            skill_registry=registry,
            pinning_store=store_factory,
            resolved_composition=second,
        )


def test_ec8_concurrent_writers_different_tenants_first_writer_wins_per_scope() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    registry.register(_manifest("b.pack"))
    comp_a = SkillResolver(registry).resolve_composition(["a.pack"])
    comp_b = SkillResolver(registry).resolve_composition(["b.pack"])
    execution_id = mint_execution_id()
    errors: list[BaseException] = []

    def worker(tenant_id: str, composition) -> None:
        try:
            bind_resolved_skill_pack(
                tenant_id=tenant_id,
                execution_id=execution_id,
                skill_profile=SkillProfile(enabled=[composition.pack.skill_ids[0]]),
                skill_registry=registry,
                pinning_store=store,
                resolved_composition=composition,
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("tenant-a", comp_a)),
        threading.Thread(target=worker, args=("tenant-b", comp_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert recover_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store,
    ).skill_ids == ("a.pack",)
    assert recover_bound_skill_pack(
        tenant_id="tenant-b",
        execution_id=execution_id,
        pinning_store=store,
    ).skill_ids == ("b.pack",)


def test_ec8_resumed_execution_mismatched_tenant_identity_rejected() -> None:
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="no bound skill pack"):
            resolve_bound_skill_pack(
                tenant_id="tenant-b",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=pinning,
                skill_execution_recovery_required=True,
            )
    finally:
        reset_active_execution_identity(token)


def test_ec8_forged_persisted_record_wrong_execution_scope_rejected() -> None:
    backing = InMemoryKVStore()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_a = mint_execution_id()
    execution_b = mint_execution_id()
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_a,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=KvSkillExecutionPinningStore(InMemoryKVStore()),
        resolved_composition=composition,
    )
    backing.set(
        "tenant-a",
        f"skill_execution_pinning:{execution_b}",
        encode_skill_execution_binding(binding),
    )
    with pytest.raises(SkillExecutionPinningStorageError, match="execution_id mismatch"):
        store.get(tenant_id="tenant-a", execution_id=execution_b)


def test_ec8_corrupted_storage_fail_closed() -> None:
    backing = InMemoryKVStore()
    backing.set("tenant-a", "skill_execution_pinning:bad", b"not-json")
    store = KvSkillExecutionPinningStore(backing)
    with pytest.raises(SkillExecutionPinningStorageError, match="invalid skill execution binding"):
        store.get(tenant_id="tenant-a", execution_id="bad")


def test_ec8_ack_loss_then_scope_verification_per_tenant() -> None:
    class AckLostCAS(InMemoryKVStore):
        def compare_and_set(self, tenant_id, key, expected, new_value, *, ttl_seconds=None):
            del ttl_seconds
            if expected is None and (tenant_id, key) not in self._data:
                self._data[(tenant_id, key)] = new_value
                raise RuntimeError("ack lost")
            return super().compare_and_set(tenant_id, key, expected, new_value)

    backing = AckLostCAS()
    store = KvSkillExecutionPinningStore(backing)
    registry = SkillRegistry()
    registry.register(_manifest("a.pack"))
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    profile = SkillProfile(enabled=["a.pack"])
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=profile,
        skill_registry=registry,
        pinning_store=KvSkillExecutionPinningStore(InMemoryKVStore()),
        resolved_composition=composition,
    )
    store.pin(binding)
    store.pin(binding)
    assert recover_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=store,
    ).snapshot_digest == composition.pack.snapshot_digest
    assert store.get(tenant_id="tenant-b", execution_id=execution_id) is None


# --- EC5 / EC9: historical replay trust boundary ---


def test_ec5_registry_version_replacement_preserves_pinned_historical_pack() -> None:
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.v1",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:v1"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.register_or_replace(
        _manifest("a.pack", "2.0.0", tool_ids=("tool.v2",)),
        binding=_binding_metadata("a.pack", "2.0.0", "sha256:v2"),
    )
    historical = recover_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning,
    )
    assert historical.resolved_skills[0].version == "1.0.0"
    assert historical.tool_ids == frozenset({"tool.v1"})


def test_ec9_deleted_catalog_blocks_content_authenticated_canonical_resolve() -> None:
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.unregister("a.pack")
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="catalog binding metadata unavailable"):
            resolve_bound_skill_pack(
                tenant_id="tenant-a",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=pinning,
            )
    finally:
        reset_active_execution_identity(token)


def test_ec9_forged_in_record_witness_metadata_rejected_before_execution_path() -> None:
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    binding = bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=mint_execution_id(),
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=KvSkillExecutionPinningStore(InMemoryKVStore()),
        resolved_composition=composition,
    )
    forged = dataclasses.replace(
        binding,
        registry_content_witness_digests=(("a.pack", "sha256:forged"),),
    )
    with pytest.raises(SkillResolutionError, match="witness"):
        resolve_bound_skill_pack(
            tenant_id="tenant-a",
            skill_profile=SkillProfile(enabled=["a.pack"]),
            skill_registry=registry,
            pinning_store=KvSkillExecutionPinningStore(InMemoryKVStore()),
            explicit_binding=forged,
        )


def test_ec9_structural_replay_without_registry_not_content_authenticated_execution() -> None:
    """Recovery without live catalog cannot satisfy verified canonical resolve."""
    pinning = KvSkillExecutionPinningStore(InMemoryKVStore())
    registry = SkillRegistry()
    registry.register(
        _manifest("a.pack"),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:trusted"),
    )
    composition = SkillResolver(registry).resolve_composition(["a.pack"])
    execution_id = mint_execution_id()
    bind_resolved_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        skill_profile=SkillProfile(enabled=["a.pack"]),
        skill_registry=registry,
        pinning_store=pinning,
        resolved_composition=composition,
    )
    registry.register_or_replace(
        _manifest("a.pack", "1.0.0", tool_ids=("tool.evil",)),
        binding=_binding_metadata("a.pack", "1.0.0", "sha256:substituted"),
    )
    recover_bound_skill_pack(
        tenant_id="tenant-a",
        execution_id=execution_id,
        pinning_store=pinning,
    )
    token = bind_active_execution_identity(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=execution_id,
    )
    try:
        with pytest.raises(SkillResolutionError, match="catalog witness"):
            resolve_bound_skill_pack(
                tenant_id="tenant-a",
                skill_profile=SkillProfile(enabled=["a.pack"]),
                skill_registry=registry,
                pinning_store=pinning,
            )
    finally:
        reset_active_execution_identity(token)
