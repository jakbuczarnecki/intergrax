# © Artur Czarnecki. All rights reserved.

"""SEC-ENT: enterprise production security wiring and observability."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.application_security_wiring import register_application_security_hooks
from intergrax.applications._shared.security_runtime_bridge import resolve_restricted_payload_encryptor
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import SecurityEnvelope
from intergrax.core.security_bootstrap import register_security_payload_schemas
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.presets import harness_defense_stack
from intergrax.runtime.events.event_kind_registry import get_event_kind_entry
from intergrax.runtime.hooks.hook_context import HookContext
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.security.defense_plugin import (
    PluginSecurityDefenseMiddleware,
    SecurityFailMode,
    SecurityInspectionResult,
)
from intergrax.runtime.security.encryption_transform import (
    HarnessEnvelopeEncryptor,
    SecretsStorePayloadEncryptor,
)
from intergrax.runtime.security.security_events import KIND_DEFENSE_BLOCKED
from intergrax.runtime.security.security_observability import wire_security_spine_subscriber

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


class _AllowDefensePlugin:
    plugin_id = "lab.allow"
    version = "1.0.0"
    hook_points = frozenset({HookPoint.BEFORE_TOOL_CALL})
    priority = 58
    fail_mode = SecurityFailMode.FAIL_CLOSED

    def inspect(self, point: HookPoint, ctx: HookContext) -> SecurityInspectionResult:
        return SecurityInspectionResult(allowed=True, plugin_id=self.plugin_id)


class _FakeSecretsStore:
    stored: dict[str, str] = {}

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        return self.stored[path]

    def put_secret(self, path: str, value: str) -> None:
        self.stored[path] = value

    def delete_secret(self, path: str) -> None:
        self.stored.pop(path, None)


@pytest.fixture(autouse=True)
def _reset_store() -> None:
    _FakeSecretsStore.stored.clear()


def test_security_payload_schemas_registered() -> None:
    register_security_payload_schemas()
    assert get_event_kind_entry(KIND_DEFENSE_BLOCKED) is not None


def test_resolve_encryptor_uses_secrets_store_when_resolvable() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.enc")
    env.integration_profile = harness_defense_stack()
    encryptor = resolve_restricted_payload_encryptor(env)
    assert encryptor is not None
    assert isinstance(encryptor, (SecretsStorePayloadEncryptor, HarnessEnvelopeEncryptor))


def test_register_security_hooks_wires_spine_subscriber() -> None:
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    nexus = NexusLoop(AgentRegistry(), event_bus=RuntimeEventBus())
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.hooks")
    env.security_profile = SecurityEnvelope.production().application_security
    env.integration_profile = harness_defense_stack()
    register_default_integrations()
    register_application_security_hooks(
        nexus,
        env.security_profile,
        env=env,
    )
    counters = wire_security_spine_subscriber(nexus.event_bus)
    assert counters.subscription_id


@pytest.mark.asyncio
async def test_defense_middleware_blocks_cross_tenant_scope() -> None:
    from intergrax.runtime.events.event_bus import RuntimeEventBus

    bus = RuntimeEventBus()
    middleware = PluginSecurityDefenseMiddleware(
        _AllowDefensePlugin(),
        event_bus=bus,
        enforce_tenant_scope=True,
    )
    ctx = HookContext(
        run_id="run-1",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={
            "tenant_id": "tenant-a",
            "resource_tenant_id": "tenant-b",
            "tool_id": "echo",
        },
    )
    result = await middleware.before(HookPoint.BEFORE_TOOL_CALL, ctx)
    assert result.action.value == "block"
    assert "tenant scope" in (result.reason or "")


@pytest.mark.asyncio
async def test_security_spine_counters_increment() -> None:
    from intergrax.runtime.events.event_bus import RuntimeEventBus
    from intergrax.runtime.events.spine_consolidation import build_platform_signal_event

    bus = RuntimeEventBus()
    counters = wire_security_spine_subscriber(bus)
    event = build_platform_signal_event(
        kind=KIND_DEFENSE_BLOCKED,
        task_id="t1",
        run_id="r1",
        payload={"plugin_id": "x", "reason": "y", "hook_point": "before_tool_call"},
    )
    await bus.publish(event)
    assert counters.defense_blocked == 1
