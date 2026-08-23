# © Artur Czarnecki. All rights reserved.

"""SEC-PLANES-EVOL: catalog bootstrap, spine signals, encrypt transform, EP fixture."""

from __future__ import annotations

import time

import pytest

from intergrax.contracts.data_classification import DataClassification
from intergrax.core.catalog_bootstrap import reset_tier0_catalog_bootstrap_for_tests
from intergrax.core.security_bootstrap import bootstrap_security_providers
from intergrax.integrations.registry.bootstrap import reset_default_integrations_state
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.security.defense_plugin import (
    PluginSecurityDefenseMiddleware,
    SecurityFailMode,
    SecurityInspectionResult,
)
from intergrax.runtime.security.defense_registry import (
    get_security_defense_plugin,
    reset_security_defense_registry_for_tests,
)
from intergrax.runtime.security.encryption_middleware import EncryptionEnforcementMiddleware
from intergrax.runtime.security.encryption_transform import (
    HarnessEnvelopeEncryptor,
    SecretsStorePayloadEncryptor,
)
from intergrax.runtime.security.security_events import (
    KIND_DEFENSE_BLOCKED,
    KIND_ENCRYPTION_DENIED,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _SlowDefensePlugin:
    plugin_id = "lab.slow_defense"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 59
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        time.sleep(0.25)
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)


class _BlockDefensePlugin:
    plugin_id = "lab.block_defense"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 59
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        return SecurityInspectionResult(
            allowed=False,
            reasons=["blocked"],
            plugin_id=self.plugin_id,
            hook_point=point.value,
        )


class _FakeSecretsStore:
    stored: dict[str, str] = {}

    def put_secret(self, path: str, value: str) -> None:
        self.stored[path] = value


@pytest.fixture(autouse=True)
def _reset_defense_registry() -> None:
    reset_security_defense_registry_for_tests()
    _FakeSecretsStore.stored.clear()


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    reset_tier0_catalog_bootstrap_for_tests()
    yield
    clear_catalog()
    reset_default_integrations_state()
    reset_tier0_catalog_bootstrap_for_tests()


@pytest.mark.usefixtures("security_defense_fixture_installed")
def test_security_bootstrap_discovers_security_defense_entry_point() -> None:
    result = bootstrap_security_providers(discover_entry_points=True)
    assert result.entry_point_plugins >= 1
    assert get_security_defense_plugin("fixture_ep.defense") is not None


@pytest.mark.asyncio
async def test_defense_blocked_emits_platform_signal() -> None:
    bus = RuntimeEventBus()
    middleware = PluginSecurityDefenseMiddleware(_BlockDefensePlugin(), event_bus=bus)
    ctx = HookContext(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={"tool_id": "echo"},
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action.value == "block"
    kinds = [event.event_kind for event in bus.history]
    assert KIND_DEFENSE_BLOCKED in kinds


@pytest.mark.asyncio
async def test_encryption_denied_emits_platform_signal() -> None:
    bus = RuntimeEventBus()
    middleware = EncryptionEnforcementMiddleware(
        enforcement_enabled=True,
        secrets_store_configured=False,
        event_bus=bus,
    )
    ctx = HookContext(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={"value": {"data_classification": DataClassification.RESTRICTED.value, "secret": "x"}},
    )
    result = await middleware.before(HookPoint.BEFORE_MEMORY_WRITE, ctx)
    assert result.action.value == "block"
    kinds = [event.event_kind for event in bus.history]
    assert KIND_ENCRYPTION_DENIED in kinds


@pytest.mark.asyncio
async def test_encryption_middleware_transforms_restricted_payload() -> None:
    store = _FakeSecretsStore()
    middleware = EncryptionEnforcementMiddleware(
        enforcement_enabled=True,
        secrets_store_configured=True,
        encryptor=SecretsStorePayloadEncryptor(store),
    )
    ctx = HookContext(
        run_id="run-enc",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={
            "value": {"data_classification": "restricted", "secret": "top-secret"},
        },
    )
    result = await middleware.before(HookPoint.BEFORE_MEMORY_WRITE, ctx)
    assert result.action.value == "modify"
    assert result.modified_payload is not None
    value = result.modified_payload["value"]
    assert isinstance(value, dict)
    assert value["__secret_ref"] == "restricted/run-enc/payload"
    assert "secret" not in value
    assert store.stored["restricted/run-enc/payload"] == "top-secret"


def test_secrets_store_encryptor_persists_and_replaces_inline_secret() -> None:
    store = _FakeSecretsStore()
    encryptor = SecretsStorePayloadEncryptor(store)
    payload = {
        "value": {"data_classification": "restricted", "secret": "vault-me"},
    }
    updated = encryptor.encrypt_payload(payload, run_id="run-ss")
    assert updated["value"]["__secret_ref"] == "restricted/run-ss/payload"
    assert store.stored["restricted/run-ss/payload"] == "vault-me"


@pytest.mark.asyncio
async def test_defense_plugin_inspection_timeout_blocks() -> None:
    middleware = PluginSecurityDefenseMiddleware(
        _SlowDefensePlugin(),
        inspection_timeout_ms=50,
    )
    ctx = HookContext(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={"tool_id": "echo"},
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action.value == "block"
    assert "timeout" in (result.reason or "")


def test_platform_security_kinds_are_domain_signals() -> None:
    from intergrax.runtime.events.spine_consolidation import get_platform_kind_entry

    defense = get_platform_kind_entry(KIND_DEFENSE_BLOCKED)
    encryption = get_platform_kind_entry(KIND_ENCRYPTION_DENIED)
    assert defense is not None
    assert encryption is not None
    assert defense.ops_hint == "ops:alert"
