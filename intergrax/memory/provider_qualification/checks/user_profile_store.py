# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    UserProfileStoreQualificationCheck,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
)
from intergrax.memory.provider_qualification.checks._helpers import failed, passed
from intergrax.memory.user_profile_memory import UserIdentity, UserPreferences, UserProfile
from intergrax.memory.user_profile_store import UserProfileStore

_CAPABILITY = MemoryProviderCapabilityKind.USER_PROFILE_STORE
_REQUIRED = MemoryProviderCheckSeverity.REQUIRED


def _qual_user_id(context: MemoryProviderQualificationContext) -> str:
    return context.user_qualification_id


def _tenant_a(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-a"


def _tenant_b(context: MemoryProviderQualificationContext) -> str:
    return f"{context.tenant_qualification_id}-b"


@dataclass(frozen=True, slots=True)
class UserProfileTenantIsolationCheck:
    check_id: str = "user_profile.tenant_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        user_id = _qual_user_id(context)
        marker = f"qual-marker-{context.qualification_run_id}"
        profile_a = UserProfile(
            identity=UserIdentity(user_id=user_id),
            preferences=UserPreferences(),
            system_instructions=marker,
        )
        await store.save_profile(tenant_id=_tenant_a(context), profile=profile_a)
        loaded_b = await store.get_profile(tenant_id=_tenant_b(context), user_id=user_id)
        if (loaded_b.system_instructions or "") == marker:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.TENANT_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class UserProfileUserIsolationCheck:
    check_id: str = "user_profile.user_isolation"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        tenant = _tenant_a(context)
        user_a = f"{context.user_qualification_id}-a"
        user_b = f"{context.user_qualification_id}-b"
        marker = f"user-marker-{context.qualification_run_id}"
        await store.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id=user_a),
                preferences=UserPreferences(),
                system_instructions=marker,
            ),
        )
        loaded_b = await store.get_profile(tenant_id=tenant, user_id=user_b)
        if (loaded_b.system_instructions or "") == marker:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.USER_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class UserProfileDeleteScopeCheck:
    check_id: str = "user_profile.delete_scope"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        user_id = _qual_user_id(context)
        marker = f"delete-scope-{context.qualification_run_id}"
        tenant_a = _tenant_a(context)
        tenant_b = _tenant_b(context)
        await store.save_profile(
            tenant_id=tenant_a,
            profile=UserProfile(
                identity=UserIdentity(user_id=user_id),
                preferences=UserPreferences(),
                system_instructions=marker,
            ),
        )
        await store.save_profile(
            tenant_id=tenant_b,
            profile=UserProfile(
                identity=UserIdentity(user_id=user_id),
                preferences=UserPreferences(),
                system_instructions="sibling-untouched",
            ),
        )
        await store.delete_profile(tenant_id=tenant_a, user_id=user_id)
        sibling = await store.get_profile(tenant_id=tenant_b, user_id=user_id)
        if (sibling.system_instructions or "") != "sibling-untouched":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.DELETE_ISOLATION_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class UserProfileIdempotentDeleteCheck:
    check_id: str = "user_profile.idempotent_delete"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        tenant = _tenant_a(context)
        user_id = _qual_user_id(context)
        await store.save_profile(
            tenant_id=tenant,
            profile=UserProfile(
                identity=UserIdentity(user_id=user_id),
                preferences=UserPreferences(),
            ),
        )
        try:
            await store.delete_profile(tenant_id=tenant, user_id=user_id)
            await store.delete_profile(tenant_id=tenant, user_id=user_id)
        except Exception as exc:
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.INVALID_FAILURE_BEHAVIOR,
                detail=type(exc).__name__,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


@dataclass(frozen=True, slots=True)
class UserProfileSaveIdempotencyCheck:
    check_id: str = "user_profile.save_idempotency"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return _CAPABILITY

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return _REQUIRED

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        store = instance
        tenant = _tenant_a(context)
        user_id = _qual_user_id(context)
        profile = UserProfile(
            identity=UserIdentity(user_id=user_id),
            preferences=UserPreferences(preferred_language="pl"),
            version=3,
        )
        await store.save_profile(tenant_id=tenant, profile=profile)
        await store.save_profile(tenant_id=tenant, profile=profile)
        loaded = await store.get_profile(tenant_id=tenant, user_id=user_id)
        if loaded.version != 3 or loaded.preferences.preferred_language != "pl":
            return failed(
                check_id=self.check_id,
                capability=_CAPABILITY,
                severity=_REQUIRED,
                reason_code=MemoryProviderQualificationFailureReason.IDEMPOTENCY_FAILURE,
            )
        return passed(check_id=self.check_id, capability=_CAPABILITY, severity=_REQUIRED)


USER_PROFILE_STORE_CHECKS: tuple[UserProfileStoreQualificationCheck, ...] = (
    UserProfileTenantIsolationCheck(),
    UserProfileUserIsolationCheck(),
    UserProfileDeleteScopeCheck(),
    UserProfileIdempotentDeleteCheck(),
    UserProfileSaveIdempotencyCheck(),
)


def default_user_profile_checks() -> tuple[UserProfileStoreQualificationCheck, ...]:
    return USER_PROFILE_STORE_CHECKS
