# © Artur Czarnecki. All rights reserved.

"""SEC-ENT: enterprise production security wiring and observability."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.applications._shared.application_security_wiring import register_application_security_hooks
from intergrax.applications._shared.security_runtime_bridge import (
    RestrictedPayloadEncryptorResolutionError,
    resolve_restricted_payload_encryptor,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.bundles import SecurityEnvelope
from intergrax.contracts.data_classification import DataClassification
from intergrax.core.security_bootstrap import register_security_payload_schemas
from intergrax.integrations.registry.bootstrap import register_default_integrations
from intergrax.integrations.registry.presets import harness_defense_stack
from intergrax.integrations.registry.profile import IntegrationProfile
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
from intergrax.runtime.security.encryption_middleware import EncryptionEnforcementMiddleware
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


def _env_with_secrets_store(store: object) -> ApplicationEnvironmentProfile:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.store")
    env.integration_profile = IntegrationProfile(secrets_store=store)
    return env


def test_resolve_encryptor_returns_none_without_secrets_store() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.none")
    assert resolve_restricted_payload_encryptor(env) is None


def test_resolve_encryptor_uses_valid_secrets_store() -> None:
    store = _FakeSecretsStore()
    encryptor = resolve_restricted_payload_encryptor(_env_with_secrets_store(store))
    assert isinstance(encryptor, SecretsStorePayloadEncryptor)
    payload = {
        "value": {"data_classification": DataClassification.RESTRICTED.value, "secret": "vault-me"},
    }
    updated = encryptor.encrypt_payload(payload, run_id="run-valid")
    value = updated["value"]
    assert isinstance(value, dict)
    assert "secret" not in value
    assert value["__secret_ref"] == "restricted/run-valid/payload"
    assert value["encryption_envelope"] == "secrets_store.v1"
    assert store.stored["restricted/run-valid/payload"] == "vault-me"


def test_resolve_encryptor_fails_closed_on_resolution_error() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.fail.resolve")
    env.integration_profile = IntegrationProfile(secrets_store="infisical")
    with patch(
        "intergrax.integrations.registry.factory.resolve_from_profile",
        side_effect=RuntimeError("secrets backend unavailable"),
    ):
        with pytest.raises(RestrictedPayloadEncryptorResolutionError) as exc_info:
            resolve_restricted_payload_encryptor(env)
    assert "failed to resolve configured secrets_store" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_resolve_encryptor_fails_closed_on_conformance_error() -> None:
    env = _env_with_secrets_store(object())
    with pytest.raises(RestrictedPayloadEncryptorResolutionError) as exc_info:
        resolve_restricted_payload_encryptor(env)
    assert "does not satisfy SecretsStore contract" in str(exc_info.value)


def test_harness_envelope_encryptor_not_selected_by_resolver() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.no.harness")
    env.integration_profile = IntegrationProfile(secrets_store="infisical")
    with patch(
        "intergrax.integrations.registry.factory.resolve_from_profile",
        side_effect=RuntimeError("resolution failed"),
    ):
        with pytest.raises(RestrictedPayloadEncryptorResolutionError):
            resolve_restricted_payload_encryptor(env)


@pytest.mark.asyncio
async def test_encryption_middleware_transforms_with_valid_secrets_store() -> None:
    store = _FakeSecretsStore()
    middleware = EncryptionEnforcementMiddleware(
        enforcement_enabled=True,
        secrets_store_configured=True,
        encryptor=SecretsStorePayloadEncryptor(store),
    )
    ctx = HookContext(
        run_id="run-mw",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={
            "value": {"data_classification": DataClassification.RESTRICTED.value, "secret": "top-secret"},
        },
    )
    result = await middleware.before(HookPoint.BEFORE_MEMORY_WRITE, ctx)
    assert result.action.value == "modify"
    assert result.modified_payload is not None
    value = result.modified_payload["value"]
    assert isinstance(value, dict)
    assert "secret" not in value
    assert value["__secret_ref"] == "restricted/run-mw/payload"
    assert store.stored["restricted/run-mw/payload"] == "top-secret"


@pytest.mark.asyncio
async def test_encryption_middleware_leaves_non_restricted_payload_untouched() -> None:
    middleware = EncryptionEnforcementMiddleware(
        enforcement_enabled=True,
        secrets_store_configured=True,
        encryptor=SecretsStorePayloadEncryptor(_FakeSecretsStore()),
    )
    ctx = HookContext(
        run_id="run-public",
        task_id="task-1",
        agent_id="agent-1",
        runtime_state={"value": {"data_classification": DataClassification.PUBLIC.value, "note": "ok"}},
    )
    result = await middleware.before(HookPoint.BEFORE_MEMORY_WRITE, ctx)
    assert result.action.value == "allow"


def test_harness_envelope_encryptor_available_for_explicit_lab_use() -> None:
    encryptor = HarnessEnvelopeEncryptor()
    payload = {
        "value": {"data_classification": DataClassification.RESTRICTED.value, "secret": "lab-only"},
    }
    updated = encryptor.encrypt_payload(payload, run_id="run-lab")
    value = updated["value"]
    assert isinstance(value, dict)
    assert value["encryption_envelope"] == "harness.v1"
    assert "secret" not in value


def test_resolve_encryptor_uses_secrets_store_when_resolvable() -> None:
    register_default_integrations()
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="sec.ent.enc")
    env.integration_profile = harness_defense_stack()
    encryptor = resolve_restricted_payload_encryptor(env)
    assert encryptor is not None
    assert isinstance(encryptor, SecretsStorePayloadEncryptor)


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
