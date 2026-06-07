# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.6 P6 catalog adapters for new integration categories."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from intergrax.integrations._shared.health import probe_client_health
from intergrax.integrations.contracts.base import HealthStatus
from intergrax.integrations.contracts.billing_meter import BillingMeterBackend, MeterEvent
from intergrax.integrations.contracts.crm import CrmAccount, CrmBackend, CrmContact, CrmTicket
from intergrax.integrations.contracts.identity_provider import IdentityProviderBackend, IdentityTenant, IdentityUser
from intergrax.integrations.contracts.ml_inference_host import InferencePrediction, MlInferenceHostBackend
from intergrax.integrations.contracts.sandbox_host import (
    SandboxArtifact,
    SandboxExecResult,
    SandboxHostBackend,
    SandboxSession,
)
from intergrax.integrations.contracts.security_scanner import ScanFinding, ScanReport, SecurityScannerBackend
from intergrax.integrations.contracts.speech_provider import (
    SpeechProviderBackend,
    SpeechSynthesisResult,
    SpeechTranscriptionResult,
)
from intergrax.integrations.contracts.vision_serving import VisionInferenceResult, VisionPrediction, VisionServingBackend
from intergrax.integrations.contracts.workflow_orchestrator import (
    WorkflowOrchestratorBackend,
    WorkflowRunHandle,
    WorkflowRunStatus,
)
from intergrax.speech_adapters.contracts.io import SpeechSynthesizeInput, SpeechTranscribeInput
from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter


def _normalize_findings(rows: object) -> list[ScanFinding]:
    findings: list[ScanFinding] = []
    for item in list(rows or []):
        if isinstance(item, ScanFinding):
            findings.append(item)
            continue
        if not isinstance(item, dict):
            continue
        findings.append(
            ScanFinding(
                id=str(item.get("id") or item.get("ID") or ""),
                severity=str(item.get("severity") or item.get("Severity") or ""),
                title=str(item.get("title") or item.get("Title") or item.get("message") or ""),
                resource=str(item.get("resource") or item.get("Target") or ""),
                detail=str(item.get("detail") or item.get("Description") or ""),
            )
        )
    return findings


class HttpSecurityScannerBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def scan_image(self, image_ref: str) -> ScanReport:
        payload = self._client.scan_image(image_ref)
        if isinstance(payload, ScanReport):
            return payload
        data = dict(payload or {})
        return ScanReport(
            target=image_ref,
            status=str(data.get("status") or "completed"),
            findings=_normalize_findings(data.get("findings") or data.get("Results") or []),
            metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
        )

    def scan_repo(self, repo_path: str) -> ScanReport:
        payload = self._client.scan_repo(repo_path)
        if isinstance(payload, ScanReport):
            return payload
        data = dict(payload or {})
        return ScanReport(
            target=repo_path,
            status=str(data.get("status") or "completed"),
            findings=_normalize_findings(data.get("findings") or data.get("results") or []),
            metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
        )

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpSandboxHostBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def create_session(self) -> SandboxSession:
        payload = self._client.create_session()
        if isinstance(payload, SandboxSession):
            return payload
        data = dict(payload or {})
        return SandboxSession(
            session_id=str(data.get("session_id") or data.get("id") or ""),
            status=str(data.get("status") or "running"),
            metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
        )

    def exec(self, session_id: str, command: str) -> SandboxExecResult:
        payload = self._client.exec(session_id, command)
        if isinstance(payload, SandboxExecResult):
            return payload
        data = dict(payload or {})
        return SandboxExecResult(
            exit_code=int(data.get("exit_code") or data.get("exitCode") or 0),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
        )

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> SandboxArtifact:
        payload = self._client.upload_artifact(session_id, local_path=local_path, remote_name=remote_name)
        if isinstance(payload, SandboxArtifact):
            return payload
        data = dict(payload or {})
        return SandboxArtifact(
            artifact_id=str(data.get("artifact_id") or data.get("id") or remote_name),
            uri=str(data.get("uri") or data.get("url") or ""),
            size_bytes=int(data.get("size_bytes") or data.get("size") or 0),
        )

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpIdentityProviderBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def verify_token(self, token: str) -> IdentityUser:
        payload = self._client.verify_token(token)
        return self._to_user(payload)

    def userinfo(self, token: str) -> IdentityUser:
        payload = self._client.userinfo(token)
        return self._to_user(payload)

    def list_tenants(self, *, limit: int = 50) -> Sequence[IdentityTenant]:
        rows = self._client.list_tenants(limit=limit)
        tenants: list[IdentityTenant] = []
        for item in list(rows or [])[:limit]:
            if isinstance(item, IdentityTenant):
                tenants.append(item)
                continue
            data = dict(item or {})
            tenants.append(
                IdentityTenant(
                    tenant_id=str(data.get("tenant_id") or data.get("id") or ""),
                    name=str(data.get("name") or ""),
                    metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
                )
            )
        return tenants

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)

    def _to_user(self, payload: object) -> IdentityUser:
        if isinstance(payload, IdentityUser):
            return payload
        data = dict(payload or {})
        return IdentityUser(
            user_id=str(data.get("user_id") or data.get("sub") or data.get("id") or ""),
            email=str(data.get("email") or ""),
            name=str(data.get("name") or data.get("preferred_username") or ""),
            tenant_id=str(data.get("tenant_id") or data.get("org_id") or ""),
            metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
        )


class SpeechAdapterBackend:
    """Bridge ``speech_adapters.SpeechAdapter`` to ``SpeechProviderBackend``."""

    def __init__(self, adapter: SpeechAdapter, *, slug: str) -> None:
        self._adapter = adapter
        self._slug = slug

    def synthesize(self, text: str, *, voice_id: str = "default") -> SpeechSynthesisResult:
        output = self._adapter.synthesize(SpeechSynthesizeInput(text=text, voice_id=voice_id))
        return SpeechSynthesisResult(audio_uri=output.audio_uri, character_count=output.character_count)

    def transcribe(self, audio_uri: str) -> SpeechTranscriptionResult:
        output = self._adapter.transcribe(SpeechTranscribeInput(audio_uri=audio_uri))
        return SpeechTranscriptionResult(transcript=output.transcript, duration_ms=output.duration_ms)

    def health(self) -> HealthStatus:
        try:
            self._adapter.validate()
        except Exception as exc:  # noqa: BLE001 — health probe surface
            return HealthStatus(slug=self._slug, healthy=False, detail=str(exc))
        return HealthStatus(slug=self._slug, healthy=True, detail="speech adapter validated")


class HttpSpeechProviderBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def synthesize(self, text: str, *, voice_id: str = "default") -> SpeechSynthesisResult:
        payload = self._client.synthesize(text, voice_id=voice_id)
        if isinstance(payload, SpeechSynthesisResult):
            return payload
        data = dict(payload or {})
        return SpeechSynthesisResult(
            audio_uri=str(data.get("audio_uri") or data.get("url") or ""),
            character_count=int(data.get("character_count") or len(text)),
        )

    def transcribe(self, audio_uri: str) -> SpeechTranscriptionResult:
        payload = self._client.transcribe(audio_uri)
        if isinstance(payload, SpeechTranscriptionResult):
            return payload
        data = dict(payload or {})
        return SpeechTranscriptionResult(
            transcript=str(data.get("transcript") or data.get("text") or ""),
            duration_ms=int(data.get("duration_ms") or 0),
        )

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpWorkflowOrchestratorBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str] | None = None) -> WorkflowRunHandle:
        payload = self._client.trigger_run(workflow_id, parameters=parameters or {})
        if isinstance(payload, WorkflowRunHandle):
            return payload
        data = dict(payload or {})
        return WorkflowRunHandle(
            run_id=str(data.get("run_id") or data.get("id") or ""),
            status=str(data.get("status") or "pending"),
            url=str(data.get("url") or ""),
            metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
        )

    def poll_status(self, run_id: str) -> WorkflowRunStatus:
        payload = self._client.poll_status(run_id)
        if isinstance(payload, WorkflowRunStatus):
            return payload
        data = dict(payload or {})
        return WorkflowRunStatus(
            run_id=run_id,
            status=str(data.get("status") or ""),
            conclusion=str(data.get("conclusion") or ""),
            logs_uri=str(data.get("logs_uri") or data.get("logs_url") or ""),
        )

    def fetch_logs(self, run_id: str, *, tail_lines: int = 200) -> str:
        return str(self._client.fetch_logs(run_id, tail_lines=tail_lines))

    def list_runs(
        self,
        *,
        workflow_id: str = "",
        limit: int = 20,
    ) -> Sequence[WorkflowRunHandle]:
        try:
            rows = self._client.list_runs(workflow_id=workflow_id, limit=limit)
        except AttributeError:
            return []
        runs: list[WorkflowRunHandle] = []
        for item in list(rows or [])[:limit]:
            if isinstance(item, WorkflowRunHandle):
                runs.append(item)
                continue
            data = dict(item or {})
            runs.append(
                WorkflowRunHandle(
                    run_id=str(data.get("run_id") or data.get("id") or ""),
                    status=str(data.get("status") or ""),
                    url=str(data.get("url") or ""),
                    metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
                )
            )
        return runs

    def cancel_run(self, run_id: str) -> WorkflowRunStatus:
        try:
            payload = self._client.cancel_run(run_id)
        except AttributeError as exc:
            raise RuntimeError("cancel_run_not_supported") from exc
        if isinstance(payload, WorkflowRunStatus):
            return payload
        data = dict(payload or {})
        return WorkflowRunStatus(
            run_id=run_id,
            status=str(data.get("status") or "cancelled"),
            conclusion=str(data.get("conclusion") or ""),
            logs_uri=str(data.get("logs_uri") or data.get("logs_url") or ""),
        )

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpVisionServingBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def predict(self, model_name: str, *, input_uri: str) -> VisionInferenceResult:
        payload = self._client.predict(model_name, input_uri=input_uri)
        if isinstance(payload, VisionInferenceResult):
            return payload
        data = dict(payload or {})
        preds: list[VisionPrediction] = []
        for item in list(data.get("predictions") or data.get("outputs") or []):
            if isinstance(item, VisionPrediction):
                preds.append(item)
                continue
            row = dict(item or {})
            preds.append(
                VisionPrediction(
                    label=str(row.get("label") or row.get("name") or ""),
                    score=float(row.get("score") or row.get("confidence") or 0.0),
                    metadata={k: str(v) for k, v in dict(row.get("metadata") or {}).items()},
                )
            )
        return VisionInferenceResult(model_name=model_name, predictions=preds)

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpMlInferenceHostBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def predict(self, model_ref: str, *, inputs: Mapping[str, Any]) -> InferencePrediction:
        payload = self._client.predict(model_ref, inputs=dict(inputs))
        if isinstance(payload, InferencePrediction):
            return payload
        if isinstance(payload, dict):
            return InferencePrediction(output=dict(payload.get("output") or payload), metadata={})
        return InferencePrediction(output={"result": payload})

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpBillingMeterBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def list_meter_events(self, *, customer_id: str, limit: int = 50) -> Sequence[MeterEvent]:
        rows = self._client.list_meter_events(customer_id=customer_id, limit=limit)
        events: list[MeterEvent] = []
        for item in list(rows or [])[:limit]:
            if isinstance(item, MeterEvent):
                events.append(item)
                continue
            data = dict(item or {})
            events.append(
                MeterEvent(
                    event_id=str(data.get("event_id") or data.get("id") or ""),
                    customer_id=str(data.get("customer_id") or customer_id),
                    metric=str(data.get("metric") or ""),
                    quantity=float(data.get("quantity") or 0.0),
                    metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
                )
            )
        return events

    def submit_meter_event(self, *, customer_id: str, metric: str, quantity: float) -> MeterEvent:
        payload = self._client.submit_meter_event(
            customer_id=customer_id,
            metric=metric,
            quantity=quantity,
        )
        if isinstance(payload, MeterEvent):
            return payload
        data = dict(payload or {})
        return MeterEvent(
            event_id=str(data.get("event_id") or data.get("id") or ""),
            customer_id=customer_id,
            metric=metric,
            quantity=quantity,
        )

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)


class HttpCrmBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def get_account(self, account_id: str) -> CrmAccount:
        payload = self._client.get_account(account_id)
        if isinstance(payload, CrmAccount):
            return payload
        data = dict(payload or {})
        return CrmAccount(
            account_id=str(data.get("account_id") or data.get("id") or account_id),
            name=str(data.get("name") or ""),
            industry=str(data.get("industry") or ""),
            metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
        )

    def list_contacts(self, *, account_id: str, limit: int = 50) -> Sequence[CrmContact]:
        rows = self._client.list_contacts(account_id=account_id, limit=limit)
        contacts: list[CrmContact] = []
        for item in list(rows or [])[:limit]:
            if isinstance(item, CrmContact):
                contacts.append(item)
                continue
            data = dict(item or {})
            contacts.append(
                CrmContact(
                    contact_id=str(data.get("contact_id") or data.get("id") or ""),
                    email=str(data.get("email") or ""),
                    name=str(data.get("name") or ""),
                    account_id=str(data.get("account_id") or account_id),
                    metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
                )
            )
        return contacts

    def list_tickets(self, *, account_id: str, limit: int = 50) -> Sequence[CrmTicket]:
        rows = self._client.list_tickets(account_id=account_id, limit=limit)
        tickets: list[CrmTicket] = []
        for item in list(rows or [])[:limit]:
            if isinstance(item, CrmTicket):
                tickets.append(item)
                continue
            data = dict(item or {})
            tickets.append(
                CrmTicket(
                    ticket_id=str(data.get("ticket_id") or data.get("id") or ""),
                    subject=str(data.get("subject") or data.get("title") or ""),
                    status=str(data.get("status") or ""),
                    account_id=str(data.get("account_id") or account_id),
                    metadata={k: str(v) for k, v in dict(data.get("metadata") or {}).items()},
                )
            )
        return tickets

    def health(self) -> HealthStatus:
        return probe_client_health(self._client, slug=self._provider)
