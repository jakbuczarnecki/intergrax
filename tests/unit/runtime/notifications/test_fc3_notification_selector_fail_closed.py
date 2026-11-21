# © Artur Czarnecki. All rights reserved.

"""FC-3 Part C: notification backend selector fail-closed (NT-FC3-N1..N13)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from intergrax.applications._shared.notification_wiring import resolve_host_notification_adapter
from intergrax.applications.contracts.application_host import ApplicationProfile
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn
from intergrax.integrations.providers.notification_channel.log.integration import LogNotificationChannelIntegration
from intergrax.integrations.registry.bootstrap import (
    register_default_integrations,
    reset_default_integrations_state,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.human.models import EscalationOutcome, EscalationTarget
from intergrax.runtime.long_running.coordinator import LongRunningCoordinator
from intergrax.runtime.notifications.backend_contract import (
    NotificationBackend,
    NotificationSettings,
    resolve_notification_settings,
)
from intergrax.runtime.notifications.factory import (
    create_notification_adapter,
    resolve_notification_adapter,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = pytest.mark.unit

_LAB_OPT_IN = LabSelectorOptIn(
    selector=NotificationBackend.LOG.value,
    reason="FC-3 notification unit negative-test positive control",
)


@pytest.fixture(autouse=True)
def _bootstrap_notification_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    register_default_integrations(preset="full")
    yield
    clear_catalog()
    reset_default_integrations_state()


def _long_running_task(*, notify_channel: str | None = None) -> Task:
    return Task(
        tenant_id="tenant-a",
        user_id="user-a",
        message="notify probe",
        state=TaskState.RUNNING,
        context=TaskContext(capability="echo.basic"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, notify_channel=notify_channel),
        ),
    )


def test_nt_fc3_n1_resolve_notification_settings_env_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_NOTIFICATION_BACKEND", raising=False)
    with pytest.raises(IntegrationConfigurationError):
        resolve_notification_settings()


def test_nt_fc3_n2_typo_raises_with_and_without_lab_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INTERGRAX_NOTIFICATION_BACKEND", "slakc")
    with pytest.raises(IntegrationConfigurationError):
        resolve_notification_settings()
    with pytest.raises(IntegrationConfigurationError):
        resolve_notification_settings(lab_opt_in=_LAB_OPT_IN)


def test_nt_fc3_n3_notification_settings_requires_backend() -> None:
    with pytest.raises(TypeError):
        NotificationSettings()  # type: ignore[call-arg]


def test_nt_fc3_n4_unsupported_enum_member_raises() -> None:
    unsupported = MagicMock(spec=NotificationBackend)
    unsupported.__eq__ = lambda self, other: False  # type: ignore[method-assign, assignment]
    settings = NotificationSettings(backend=NotificationBackend.LOG)
    object.__setattr__(settings, "backend", unsupported)
    with pytest.raises(IntegrationConfigurationError, match="unsupported notification backend"):
        create_notification_adapter(settings)


def test_nt_fc3_n5_resolve_adapter_unknown_channel() -> None:
    with pytest.raises(IntegrationConfigurationError):
        resolve_notification_adapter("lgo")


def test_nt_fc3_n6_resolve_adapter_none_without_opt_in() -> None:
    with pytest.raises(IntegrationConfigurationError):
        resolve_notification_adapter(None)


def test_nt_fc3_n7_resolve_adapter_none_with_opt_in() -> None:
    adapter = resolve_notification_adapter(None, lab_opt_in=_LAB_OPT_IN)
    assert isinstance(adapter, LogNotificationChannelIntegration)


@pytest.mark.parametrize(
    "method_name",
    [
        "notify_progress",
        "notify_partial_result",
        "notify_hitl_pause",
        "notify_escalation",
    ],
)
@pytest.mark.asyncio
async def test_nt_fc3_n8_coordinator_refuses_without_adapter_or_channel(
    method_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _long_running_task()
    with patch(
        "intergrax.integrations.providers.notification_channel.log.bundle.create_log_notification_channel",
    ) as log_mock:
        with pytest.raises(IntegrationConfigurationError):
            if method_name == "notify_progress":
                await LongRunningCoordinator.notify_progress(
                    task,
                    subject="s",
                    body="b",
                )
            elif method_name == "notify_partial_result":
                await LongRunningCoordinator.notify_partial_result(
                    task,
                    progress_message="p",
                )
            elif method_name == "notify_hitl_pause":
                await LongRunningCoordinator.notify_hitl_pause(
                    task,
                    progress_message="p",
                )
            else:
                await LongRunningCoordinator.notify_escalation(
                    task,
                    outcome=EscalationOutcome(target=EscalationTarget.HUMAN_OPERATOR, level=1),
                    progress_message="p",
                )
        log_mock.assert_not_called()


@pytest.mark.asyncio
async def test_nt_fc3_n9_unknown_channel_without_adapter() -> None:
    task = _long_running_task(notify_channel="lgo")
    with pytest.raises(IntegrationConfigurationError):
        await LongRunningCoordinator.notify_escalation(
            task,
            outcome=EscalationOutcome(target=EscalationTarget.HUMAN_OPERATOR, level=1),
            progress_message="p",
        )


@pytest.mark.asyncio
async def test_nt_fc3_n10_injected_adapter_receives_message() -> None:
    task = _long_running_task()
    injected = MagicMock()
    injected.notify = AsyncMock()

    await LongRunningCoordinator.notify_escalation(
        task,
        outcome=EscalationOutcome(target=EscalationTarget.HUMAN_OPERATOR, level=1),
        progress_message="done",
        adapter=injected,
    )
    injected.notify.assert_awaited_once()  # type: ignore[attr-defined]


def test_nt_fc3_n11_strict_host_without_channel_binding() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.notification.strict")
    env = env.model_copy(
        update={"meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})},
    )
    with pytest.raises(IntegrationConfigurationError):
        resolve_host_notification_adapter(env, integration_profile=IntegrationProfile())


def test_nt_fc3_n12_product_host_without_channel_binding() -> None:
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="fc3.notification.product")
    with pytest.raises(IntegrationConfigurationError):
        resolve_host_notification_adapter(env, integration_profile=IntegrationProfile())


def test_nt_fc3_n13_lab_derives_log_product_uses_profile_adapter() -> None:
    env_lab = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.notification.lab")
    assert env_lab.application_profile is ApplicationProfile.LAB
    lab_adapter = resolve_host_notification_adapter(env_lab, integration_profile=IntegrationProfile())
    assert isinstance(lab_adapter, LogNotificationChannelIntegration)

    env_product = ApplicationEnvironmentProfile.product_defaults(profile_id="fc3.notification.product.bound")
    bound = IntegrationProfile.lab()
    product_adapter = resolve_host_notification_adapter(env_product, integration_profile=bound)
    assert isinstance(product_adapter, LogNotificationChannelIntegration)
