# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13R typed qualification and boundary tests."""

from __future__ import annotations

import ast
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.memory.contracts.entity_temporal_memory import EntityTemporalMemoryStore
from intergrax.memory.contracts.provider_qualification import (
    MemoryProviderCapabilityKind,
    MemoryProviderCheckResult,
    MemoryProviderCheckSeverity,
    MemoryProviderDescriptor,
    MemoryProviderQualificationContext,
    MemoryProviderQualificationFailureReason,
    MemoryProviderQualificationRequest,
    MemoryProviderQualificationStatus,
    validate_memory_provider_descriptor,
    validate_memory_provider_qualification_request,
)
from intergrax.memory.provider_qualification import (
    MemoryProviderCapabilityFactories,
    MemoryProviderInstanceFactory,
    MemoryProviderQualificationRunner,
)
from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_QUAL_ROOT = _REPO / "intergrax" / "memory" / "provider_qualification"
_CONTRACT = _REPO / "intergrax" / "memory" / "contracts" / "provider_qualification.py"
_CORE_FILES = (
    _CONTRACT,
    _QUAL_ROOT / "runner.py",
    _QUAL_ROOT / "bindings.py",
)


def _context(run_id: str = "run-typed") -> MemoryProviderQualificationContext:
    return MemoryProviderQualificationContext(
        qualification_run_id=run_id,
        tenant_qualification_id="qual-tenant",
        user_qualification_id="qual-user",
        workspace_qualification_id="qual-workspace",
        reference_time_iso="2025-01-01T00:00:00+00:00",
    )


@dataclass(slots=True)
class _StaticFactory(MemoryProviderInstanceFactory[UserProfileStore]):
    _supplier: Callable[[], UserProfileStore]
    _disposer: Callable[[UserProfileStore], Awaitable[None]] | None = None

    async def create(self) -> UserProfileStore:
        return self._supplier()

    async def dispose(self, instance: UserProfileStore) -> None:
        if self._disposer is not None:
            await self._disposer(instance)


class _BrokenCreateFactory(MemoryProviderInstanceFactory[UserProfileStore]):
    async def create(self) -> UserProfileStore:
        raise RuntimeError("plugin materialization failed")

    async def dispose(self, instance: UserProfileStore) -> None:
        return None


class _BrokenDisposeFactory(MemoryProviderInstanceFactory[UserProfileStore]):
    async def create(self) -> UserProfileStore:
        return InMemoryUserProfileStore()

    async def dispose(self, instance: UserProfileStore) -> None:
        raise RuntimeError("dispose failed")


@dataclass(frozen=True, slots=True)
class _TypedUserProfileCheck:
    check_id: str = "typed.user_profile.marker"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return MemoryProviderCapabilityKind.USER_PROFILE_STORE

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return MemoryProviderCheckSeverity.OPTIONAL

    async def run(
        self,
        instance: UserProfileStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        assert isinstance(instance, InMemoryUserProfileStore)
        return MemoryProviderCheckResult(
            check_id=self.check_id,
            capability=self.capability,
            severity=self.severity,
            passed=True,
        )


@dataclass(frozen=True, slots=True)
class _TypedEntityCheck:
    check_id: str = "typed.entity.marker"

    @property
    def capability(self) -> MemoryProviderCapabilityKind:
        return MemoryProviderCapabilityKind.ENTITY_TEMPORAL_MEMORY_STORE

    @property
    def severity(self) -> MemoryProviderCheckSeverity:
        return MemoryProviderCheckSeverity.OPTIONAL

    async def run(
        self,
        instance: EntityTemporalMemoryStore,
        context: MemoryProviderQualificationContext,
    ) -> MemoryProviderCheckResult:
        return MemoryProviderCheckResult(
            check_id=self.check_id,
            capability=self.capability,
            severity=self.severity,
            passed=True,
        )


def _annotation_uses_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Name) and node.id == name:
        return True
    if isinstance(node, ast.Subscript):
        return _annotation_uses_name(node.value, name)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_uses_name(node.left, name) or _annotation_uses_name(node.right, name)
    if isinstance(node, ast.Tuple):
        return any(_annotation_uses_name(elt, name) for elt in node.elts)
    return False


def _collect_annotation_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            if _annotation_uses_name(node.annotation, "object"):
                names.add("object")
            if _annotation_uses_name(node.annotation, "Any"):
                names.add("Any")
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            if _annotation_uses_name(node.annotation, "object"):
                names.add("object")
            if _annotation_uses_name(node.annotation, "Any"):
                names.add("Any")
    return names


def _module_imports_full(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_descriptor_validation_rejects_empty_provider_id() -> None:
    with pytest.raises(ValueError, match="provider_id"):
        validate_memory_provider_descriptor(
            MemoryProviderDescriptor(provider_id="   ", capabilities=())
        )


def test_descriptor_validation_rejects_duplicate_capabilities() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        validate_memory_provider_descriptor(
            MemoryProviderDescriptor(
                provider_id="dup",
                capabilities=(
                    MemoryProviderCapabilityKind.USER_PROFILE_STORE,
                    MemoryProviderCapabilityKind.USER_PROFILE_STORE,
                ),
            )
        )


def test_request_validation_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="both required_capabilities and optional"):
        validate_memory_provider_qualification_request(
            MemoryProviderQualificationRequest(
                required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
                optional_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            )
        )


@pytest.mark.asyncio
async def test_descriptor_authority_skips_factory_materialization() -> None:
    created = False

    def _create() -> UserProfileStore:
        nonlocal created
        created = True
        return InMemoryUserProfileStore()

    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(provider_id="hidden.factory", capabilities=()),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(_create),
        ),
    )
    assert created is False
    cap = result.capability_results[0]
    assert cap.status is MemoryProviderQualificationStatus.NOT_QUALIFIED
    assert (
        MemoryProviderQualificationFailureReason.UNSUPPORTED_CAPABILITY in cap.reason_codes
    )


@pytest.mark.asyncio
async def test_optional_descriptor_unsupported_is_not_supported() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="partial",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
            optional_capabilities=(MemoryProviderCapabilityKind.SESSION_STORAGE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(lambda: InMemoryUserProfileStore()),
        ),
    )
    session_cap = next(
        item
        for item in result.capability_results
        if item.capability is MemoryProviderCapabilityKind.SESSION_STORAGE
    )
    assert session_cap.status is MemoryProviderQualificationStatus.NOT_SUPPORTED


@pytest.mark.asyncio
async def test_materialization_failure_retains_evidence() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="broken.create",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_BrokenCreateFactory()),
    )
    cap = result.capability_results[0]
    assert cap.status is MemoryProviderQualificationStatus.BLOCKED
    assert cap.checks_executed >= 1
    assert cap.checks_failed >= 1
    materialization = next(
        item for item in cap.check_results if item.check_id == "materialization.create"
    )
    assert (
        materialization.reason_code
        is MemoryProviderQualificationFailureReason.MATERIALIZATION_FAILURE
    )


@pytest.mark.asyncio
async def test_cleanup_failure_retains_evidence() -> None:
    runner = MemoryProviderQualificationRunner()
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="broken.dispose",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(user_profile_store=_BrokenDisposeFactory()),
    )
    cap = result.capability_results[0]
    dispose = next(
        item for item in cap.check_results if item.check_id == "materialization.dispose"
    )
    assert dispose.reason_code is MemoryProviderQualificationFailureReason.CLEANUP_FAILURE
    assert cap.status is MemoryProviderQualificationStatus.NOT_QUALIFIED


@pytest.mark.asyncio
async def test_typed_custom_user_profile_check_executes() -> None:
    runner = MemoryProviderQualificationRunner(
        extra_user_profile_checks=(_TypedUserProfileCheck(),)
    )
    result = await runner.qualify(
        descriptor=MemoryProviderDescriptor(
            provider_id="typed.custom",
            capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        context=_context(),
        request=MemoryProviderQualificationRequest(
            required_capabilities=(MemoryProviderCapabilityKind.USER_PROFILE_STORE,),
        ),
        factories=MemoryProviderCapabilityFactories(
            user_profile_store=_StaticFactory(lambda: InMemoryUserProfileStore()),
        ),
    )
    ids = {item.check_id for item in result.capability_results[0].check_results}
    assert "typed.user_profile.marker" in ids


def test_qualification_core_has_no_runtime_imports() -> None:
    for path in sorted(_QUAL_ROOT.rglob("*.py")):
        modules = _module_imports_full(path)
        for module in modules:
            assert not module.startswith("intergrax.runtime"), f"{path} imports {module}"


def test_public_core_contracts_avoid_object_and_any() -> None:
    found: set[str] = set()
    for path in _CORE_FILES:
        found |= _collect_annotation_names(path)
    assert "object" not in found
    assert "Any" not in found
