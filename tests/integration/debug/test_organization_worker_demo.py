# © Artur Czarnecki. All rights reserved.

import pytest
from fastapi.testclient import TestClient

from intergrax.lab.organization_worker import (
    ORG_WORKER_CAPABILITY,
    create_organization_worker_lab_app,
    enrich_organization_worker_task,
)
from intergrax.runtime.long_running.notification import LoggingNotificationAdapter
from intergrax.runtime.notifications.templates.hitl import HITL_PAUSE_TEMPLATE_ID
from intergrax.runtime.task.task import Task, TaskContext, TaskState


class _RecordingNotificationAdapter(LoggingNotificationAdapter):
    messages: list = []

    async def notify(self, message) -> None:
        _RecordingNotificationAdapter.messages.append(message)
        await super().notify(message)


def _slack_intake_payload(*, text: str) -> dict:
    return {
        "command": "/intergrax",
        "text": text,
        "user_id": "U_ORG_1",
        "team_id": "T_ORG_1",
        "trigger_id": "trigger_org_1",
    }


def _teams_intake_payload(*, text: str) -> dict:
    return {
        "type": "message",
        "id": "activity_org_1",
        "timestamp": "2026-05-27T10:00:00.000Z",
        "serviceUrl": "https://smba.trafficmanager.net/teams/",
        "channelId": "msteams",
        "from": {"id": "29:org_user", "name": "Org Manager", "aadObjectId": "aad-org-1"},
        "conversation": {"id": "conv_org", "tenantId": "T_ORG_TEAMS"},
        "text": f"<at>Intergrax</at> {text}",
        "entities": [
            {
                "type": "mention",
                "text": "<at>Intergrax</at>",
                "mentioned": {"id": "28:bot", "name": "Intergrax"},
            }
        ],
        "channelData": {"teamsTeamId": "team-org"},
    }


@pytest.mark.unit
@pytest.mark.gate
def test_enrich_organization_worker_task_enables_long_running():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="Acme Corp Q1",
        context=TaskContext(capability=ORG_WORKER_CAPABILITY),
        metadata={"interaction_channel": "slash_command"},
    )
    enriched = enrich_organization_worker_task(task)
    assert enriched.options.long_running.enabled is True
    assert enriched.options.long_running.checkpoint_on_pause is True
    assert enriched.options.long_running.notify_channel == "slack"


@pytest.mark.integration
@pytest.mark.gate
def test_organization_worker_demo_slack_intake_hitl_resume(tmp_path):
    _RecordingNotificationAdapter.messages = []
    app = create_organization_worker_lab_app(
        checkpoints_db_path=tmp_path / "org_worker_ckpt.db",
        notification_adapter=_RecordingNotificationAdapter(),
    )
    with TestClient(app) as client:
        paused = client.post(
            "/debug/interactions/intake",
            params={"tenant": "T_ORG_1", "execute": "true"},
            json=_slack_intake_payload(text=f"{ORG_WORKER_CAPABILITY} Acme Corp Q1"),
        )
        assert paused.status_code == 200
        paused_body = paused.json()
        assert paused_body["capability"] == ORG_WORKER_CAPABILITY
        assert paused_body["interaction_channel"] == "slash_command"
        assert paused_body["state"] == TaskState.WAITING_FOR_HUMAN.value
        assert paused_body["resume_token"]

        notification = _RecordingNotificationAdapter.messages[-1]
        assert notification.metadata.get("template") == HITL_PAUSE_TEMPLATE_ID
        assert notification.channel == "slack"
        assert "reply with `approve`" in notification.body

        resumed = client.post(
            f"/debug/tasks/{paused_body['task_id']}/human-response",
            params={"tenant": paused_body["tenant_id"]},
            json={"response": "approve", "resume_token": paused_body["resume_token"]},
        )
        assert resumed.status_code == 200
        resumed_body = resumed.json()
        assert resumed_body["state"] == TaskState.COMPLETED.value
        assert "delivered to finance channel" in (resumed_body["answer"] or "").lower()


@pytest.mark.integration
@pytest.mark.gate
def test_organization_worker_demo_teams_intake_hitl_resume(tmp_path):
    _RecordingNotificationAdapter.messages = []
    app = create_organization_worker_lab_app(
        checkpoints_db_path=tmp_path / "org_worker_teams_ckpt.db",
        notification_adapter=_RecordingNotificationAdapter(),
    )
    with TestClient(app) as client:
        paused = client.post(
            "/debug/interactions/intake",
            params={"execute": "true"},
            json=_teams_intake_payload(text=f"{ORG_WORKER_CAPABILITY} Contoso Q2"),
        )
        assert paused.status_code == 200
        paused_body = paused.json()
        assert paused_body["interaction_channel"] == "teams"
        assert paused_body["tenant_id"] == "T_ORG_TEAMS"
        assert paused_body["state"] == TaskState.WAITING_FOR_HUMAN.value

        notification = _RecordingNotificationAdapter.messages[-1]
        assert notification.channel == "teams"
        assert notification.metadata.get("template") == HITL_PAUSE_TEMPLATE_ID

        resumed = client.post(
            f"/debug/tasks/{paused_body['task_id']}/human-response",
            params={"tenant": paused_body["tenant_id"]},
            json={"response": "approve", "resume_token": paused_body["resume_token"]},
        )
        assert resumed.status_code == 200
        assert resumed.json()["state"] == TaskState.COMPLETED.value
