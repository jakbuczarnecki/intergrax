# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13C integration harness: canonical qualification + durable reopen/delete proofs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderDescriptor,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationResult,
    MemoryProviderQualificationStatus,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_store import UserProfileStore

CreateStore = Callable[[], UserProfileStore]
DisposeStore = Callable[[UserProfileStore], Awaitable[None]]


class DurabilityQualificationMode(StrEnum):
    DURABLE_PERSISTENCE = "durable_persistence"
    ADAPTER_RECREATION_ONLY = "adapter_recreation_only"
    NOT_TESTED = "not_tested"


def qualification_tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def qualification_user_id(context: MemoryProviderQualificationContext) -> str:
    return context.user_qualification_id


@dataclass(frozen=True, slots=True)
class DurableUserProfileQualificationEvidence:
    canonical: MemoryProviderQualificationResult
    durability_mode: DurabilityQualificationMode
    reopen_passed: bool | None
    delete_reopen_passed: bool | None
    adapter_recreation_passed: bool | None = None
    durability_reason: MemoryProviderQualificationFailureReason | None = None
    durability_detail: str | None = None

    @property
    def production_durable_qualified(self) -> bool:
        if self.canonical.status is not MemoryProviderQualificationStatus.QUALIFIED:
            return False
        if self.durability_mode is not DurabilityQualificationMode.DURABLE_PERSISTENCE:
            return False
        if self.reopen_passed is not True:
            return False
        if self.delete_reopen_passed is not True:
            return False
        return True


async def _prove_user_profile_cross_instance_read_fidelity(
    *,
    context: MemoryProviderQualificationContext,
    create_store: CreateStore,
    dispose_store: DisposeStore,
    tenant_suffix: str,
    user_suffix: str,
    marker_prefix: str,
) -> tuple[bool, str | None]:
    tenant = f"{context.tenant_qualification_id}{tenant_suffix}"
    user_id = f"{context.user_qualification_id}{user_suffix}"
    marker = f"{marker_prefix}-{context.qualification_run_id}"

    store = create_store()
    try:
        await store.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id=user_id),
                preferences=UserPreferences(preferred_language="pl"),
                system_instructions=marker,
            ),
        )
    finally:
        await dispose_store(store)

    reopened = create_store()
    try:
        loaded = await reopened.get_profile(tenant_id=tenant, user_id=user_id)
    finally:
        await dispose_store(reopened)

    if (loaded.system_instructions or "") != marker:
        return False, "reopen_read_fidelity_mismatch"
    if loaded.preferences.preferred_language != "pl":
        return False, "reopen_preferences_mismatch"
    return True, None


async def prove_user_profile_reopen_durability(
    *,
    context: MemoryProviderQualificationContext,
    create_store: CreateStore,
    dispose_store: DisposeStore,
) -> tuple[bool, str | None]:
    return await _prove_user_profile_cross_instance_read_fidelity(
        context=context,
        create_store=create_store,
        dispose_store=dispose_store,
        tenant_suffix="-a",
        user_suffix="",
        marker_prefix="durability-reopen",
    )


async def prove_user_profile_adapter_recreation(
    *,
    context: MemoryProviderQualificationContext,
    create_store: CreateStore,
    dispose_store: DisposeStore,
) -> tuple[bool, str | None]:
    return await _prove_user_profile_cross_instance_read_fidelity(
        context=context,
        create_store=create_store,
        dispose_store=dispose_store,
        tenant_suffix="-adapter",
        user_suffix="-adapter",
        marker_prefix="adapter-recreation",
    )


async def prove_user_profile_delete_durability(
    *,
    context: MemoryProviderQualificationContext,
    create_store: CreateStore,
    dispose_store: DisposeStore,
) -> tuple[bool, str | None]:
    tenant = f"{context.tenant_qualification_id}-dur-delete"
    user_id = f"{context.user_qualification_id}-dur-delete"
    marker = f"durability-delete-{context.qualification_run_id}"

    store = create_store()
    try:
        await store.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id=user_id),
                preferences=UserPreferences(),
                system_instructions=marker,
            ),
        )
        await store.delete_profile(tenant_id=tenant, user_id=user_id)
    finally:
        await dispose_store(store)

    reopened = create_store()
    try:
        loaded = await reopened.get_profile(tenant_id=tenant, user_id=user_id)
    finally:
        await dispose_store(reopened)

    if (loaded.system_instructions or "") == marker:
        return False, "deleted_profile_still_present_after_reopen"
    return True, None


async def run_durable_user_profile_production_qualification(
    *,
    descriptor: MemoryProviderDescriptor,
    context: MemoryProviderQualificationContext,
    request: MemoryProviderQualificationRequest,
    factories: MemoryProviderCapabilityFactories,
    create_store: CreateStore,
    dispose_store: DisposeStore,
    durability_mode: DurabilityQualificationMode = DurabilityQualificationMode.DURABLE_PERSISTENCE,
    run_delete_durability: bool = True,
    runner: MemoryProviderQualificationRunner | None = None,
) -> DurableUserProfileQualificationEvidence:
    qualification_runner = runner or MemoryProviderQualificationRunner()
    canonical = await qualification_runner.qualify(
        descriptor=descriptor,
        context=context,
        request=request,
        factories=factories,
    )

    reopen_passed: bool | None = None
    delete_reopen_passed: bool | None = None
    adapter_recreation_passed: bool | None = None
    durability_reason: MemoryProviderQualificationFailureReason | None = None
    durability_detail: str | None = None

    if durability_mode is DurabilityQualificationMode.DURABLE_PERSISTENCE:
        reopen_passed, reopen_detail = await prove_user_profile_reopen_durability(
            context=context,
            create_store=create_store,
            dispose_store=dispose_store,
        )
        if run_delete_durability:
            delete_reopen_passed, delete_detail = await prove_user_profile_delete_durability(
                context=context,
                create_store=create_store,
                dispose_store=dispose_store,
            )
        if reopen_passed is False:
            durability_reason = MemoryProviderQualificationFailureReason.DURABILITY_FAILURE
            durability_detail = reopen_detail
        elif delete_reopen_passed is False:
            durability_reason = MemoryProviderQualificationFailureReason.DURABILITY_FAILURE
            durability_detail = delete_detail
    elif durability_mode is DurabilityQualificationMode.ADAPTER_RECREATION_ONLY:
        adapter_recreation_passed, _adapter_detail = await prove_user_profile_adapter_recreation(
            context=context,
            create_store=create_store,
            dispose_store=dispose_store,
        )

    return DurableUserProfileQualificationEvidence(
        canonical=canonical,
        durability_mode=durability_mode,
        reopen_passed=reopen_passed,
        delete_reopen_passed=delete_reopen_passed,
        adapter_recreation_passed=adapter_recreation_passed,
        durability_reason=durability_reason,
        durability_detail=durability_detail,
    )


def user_profile_qualification_request() -> MemoryProviderQualificationRequest:
    return MemoryProviderQualificationRequest(
        required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
    )
