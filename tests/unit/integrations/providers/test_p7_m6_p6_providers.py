# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for Phase M.6 P6 integration providers and harness presets."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from intergrax.integrations._shared.conformance import (
    assert_billing_meter_backend,
    assert_ci_cd_backend,
    assert_crm_backend,
    assert_feature_flag_backend,
    assert_identity_provider_backend,
    assert_issue_tracker,
    assert_ml_inference_host_backend,
    assert_observability_backend,
    assert_sandbox_host_backend,
    assert_secrets_store,
    assert_security_scanner_backend,
    assert_speech_provider_backend,
    assert_vision_serving_backend,
    assert_workflow_orchestrator_backend,
)
from intergrax.integrations.contracts.base import HealthStatus, IntegrationCategory, IntegrationStatus
from intergrax.integrations.registry.bootstrap import register_default_integrations, reset_default_integrations_state
from intergrax.integrations.registry.catalog import catalog_snapshot, clear_catalog
from intergrax.integrations.registry.harness_lab_health import health_check_harness_m6_p6_probes
from intergrax.integrations.registry.harness_lab_stack import HARNESS_M6_P6_PROBE_SLUGS
from intergrax.integrations.registry import presets
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

M6_P6_SLUGS = (
    "trivy",
    "snyk",
    "semgrep",
    "infisical",
    "e2b",
    "modal",
    "daytona",
    "auth0",
    "keycloak",
    "workos",
    "argocd",
    "buildkite",
    "jenkins",
    "elevenlabs",
    "deepgram",
    "newrelic",
    "splunk",
    "zendesk",
    "statsig",
    "prefect",
    "airflow",
    "typesense",
    "neon",
    "pulsar",
    "algolia",
    "confluent",
    "backblaze_b2",
    "triton",
    "replicate",
    "stripe",
    "salesforce",
    "hubspot",
)


@pytest.fixture(autouse=True)
def _clean_catalog() -> None:
    clear_catalog()
    reset_default_integrations_state()
    yield
    clear_catalog()
    reset_default_integrations_state()


class _FakeHealthClient:
    def health(self) -> bool:
        return True


class _FakeScannerClient:
    def scan_image(self, image_ref: str) -> dict[str, Any]:
        return {"target": image_ref, "status": "completed", "findings": []}

    def scan_repo(self, repo_path: str) -> dict[str, Any]:
        return {"target": repo_path, "status": "completed", "findings": []}

    def health(self) -> bool:
        return True


class _FakeSandboxClient:
    def create_session(self) -> dict[str, str]:
        return {"session_id": "s1", "status": "running"}

    def exec(self, session_id: str, command: str) -> dict[str, Any]:
        return {"exit_code": 0, "stdout": command, "stderr": ""}

    def upload_artifact(self, session_id: str, *, local_path: str, remote_name: str) -> dict[str, str]:
        return {"artifact_id": remote_name, "uri": f"sandbox://{session_id}/{remote_name}"}

    def health(self) -> bool:
        return True


class _FakeIdentityClient:
    def verify_token(self, token: str) -> dict[str, str]:
        return {"user_id": "u1", "email": "user@example.com", "sub": token}

    def userinfo(self, token: str) -> dict[str, str]:
        return {"user_id": "u1", "email": "user@example.com"}

    def list_tenants(self, *, limit: int) -> list[dict[str, str]]:
        del limit
        return [{"tenant_id": "t1", "name": "Tenant"}]

    def health(self) -> bool:
        return True


class _FakeCiClient:
    def get_workflow_run(self, run_id: str) -> dict[str, Any]:
        return {"id": run_id, "status": "completed", "conclusion": "success"}

    def list_check_suites(self, *, ref: str, limit: int = 20) -> list[dict[str, Any]]:
        del limit
        return [{"id": "1", "name": ref, "status": "completed"}]

    def health(self) -> bool:
        return True


class _FakeSpeechClient:
    def synthesize(self, text: str, *, voice_id: str = "default") -> dict[str, Any]:
        del voice_id
        return {"audio_uri": "deepgram://audio/1", "character_count": len(text)}

    def transcribe(self, audio_uri: str) -> dict[str, str]:
        return {"transcript": audio_uri, "duration_ms": 1000}

    def health(self) -> bool:
        return True


class _FakeObsClient:
    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> float:
        del promql, eval_time
        return 1.0

    def query_range(self, promql: str, *, start: float, end: float, step: str) -> list[dict[str, float]]:
        del promql, step
        return [{"timestamp": start, "value": 1.0}]

    def health(self) -> bool:
        return True


class _FakeIssueClient:
    def get_issue(self, issue_key: str) -> dict[str, Any]:
        return {"key": issue_key, "summary": "Task", "status": "open"}

    def add_comment(self, issue_key: str, body: str) -> dict[str, Any]:
        return {"id": "c1", "body": body}

    def search_issues(self, jql: str, *, limit: int) -> list[dict[str, Any]]:
        del limit
        return [{"key": "1", "summary": jql, "status": "open"}]

    def health(self) -> bool:
        return True


class _FakeFlagClient:
    def evaluate_flag(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> dict[str, Any]:
        del tenant_id, user_id
        return {"enabled": True, "variant": flag_key}

    def health(self) -> bool:
        return True


class _FakeWorkflowClient:
    def trigger_run(self, workflow_id: str, *, parameters: dict[str, str]) -> dict[str, str]:
        del parameters
        return {"run_id": workflow_id, "status": "pending"}

    def poll_status(self, run_id: str) -> dict[str, str]:
        return {"status": "success", "conclusion": "success"}

    def fetch_logs(self, run_id: str, *, tail_lines: int) -> str:
        del tail_lines
        return f"logs for {run_id}"

    def health(self) -> bool:
        return True


class _FakeVisionClient:
    def predict(self, model_name: str, *, input_uri: str) -> dict[str, Any]:
        return {"predictions": [{"label": model_name, "score": 0.9, "input": input_uri}]}

    def health(self) -> bool:
        return True


class _FakeMlClient:
    def predict(self, model_ref: str, *, inputs: dict[str, Any]) -> dict[str, Any]:
        return {"output": {"model": model_ref, "inputs": inputs}}

    def health(self) -> bool:
        return True


class _FakeBillingClient:
    def list_meter_events(self, *, customer_id: str, limit: int) -> list[dict[str, Any]]:
        return [{"event_id": "e1", "customer_id": customer_id, "metric": "tokens", "quantity": 1.0}]

    def submit_meter_event(self, *, customer_id: str, metric: str, quantity: float) -> dict[str, Any]:
        return {"event_id": "e2", "customer_id": customer_id, "metric": metric, "quantity": quantity}

    def health(self) -> bool:
        return True


class _FakeCrmClient:
    def get_account(self, account_id: str) -> dict[str, str]:
        return {"account_id": account_id, "name": "Acme"}

    def list_contacts(self, *, account_id: str, limit: int) -> list[dict[str, str]]:
        del limit
        return [{"contact_id": "c1", "email": "a@example.com", "account_id": account_id}]

    def list_tickets(self, *, account_id: str, limit: int) -> list[dict[str, str]]:
        del limit
        return [{"ticket_id": "t1", "subject": "Help", "account_id": account_id}]

    def health(self) -> bool:
        return True


class _FakeSecretsClient:
    def read_secret(self, path: str, *, version: Optional[str] = None) -> str:
        del version
        return f"secret:{path}"

    def write_secret(self, path: str, value: str) -> None:
        del path, value

    def delete_secret(self, path: str) -> None:
        del path

    def health(self) -> bool:
        return True


@pytest.mark.parametrize("slug", M6_P6_SLUGS)
def test_m6_p6_registered_stable(slug: str) -> None:
    register_default_integrations()
    meta = catalog_snapshot()[slug]
    assert meta.status is IntegrationStatus.STABLE


def test_p7_security_scanners() -> None:
    from intergrax.integrations.providers.security_scanner.trivy.bundle import create_trivy_security_scanner
    from intergrax.integrations.providers.security_scanner.snyk.bundle import create_snyk_security_scanner
    from intergrax.integrations.providers.security_scanner.semgrep.bundle import create_semgrep_security_scanner

    for factory in (create_trivy_security_scanner, create_snyk_security_scanner, create_semgrep_security_scanner):
        backend = factory(client=_FakeScannerClient())
        assert_security_scanner_backend(backend)
        health = backend.health()
        assert isinstance(health, HealthStatus)
        assert health.healthy is True


def test_p7_sandbox_and_identity() -> None:
    from intergrax.integrations.providers.sandbox_host.e2b.bundle import create_e2b_sandbox_host
    from intergrax.integrations.providers.identity_provider.auth0.bundle import create_auth0_identity_provider

    sandbox = create_e2b_sandbox_host(client=_FakeSandboxClient())
    assert_sandbox_host_backend(sandbox)
    session = sandbox.create_session()
    assert session.session_id == "s1"

    identity = create_auth0_identity_provider(client=_FakeIdentityClient())
    assert_identity_provider_backend(identity)
    user = identity.verify_token("token")
    assert user.user_id == "u1"


def test_p7_gitops_ci_and_speech() -> None:
    from intergrax.integrations.providers.ci_cd.argocd.bundle import create_argocd_ci_cd
    from intergrax.integrations.providers.speech_provider.deepgram.bundle import create_deepgram_speech_provider

    ci = create_argocd_ci_cd(client=_FakeCiClient())
    assert_ci_cd_backend(ci)

    speech = create_deepgram_speech_provider(client=_FakeSpeechClient())
    assert_speech_provider_backend(speech)
    result = speech.synthesize("hello")
    assert result.audio_uri


def test_p7_observability_issue_flag_workflow() -> None:
    from intergrax.integrations.providers.observability_backend.newrelic.bundle import create_newrelic_observability_backend
    from intergrax.integrations.providers.issue_tracker.zendesk.bundle import create_zendesk_issue_tracker
    from intergrax.integrations.providers.feature_flag.statsig.bundle import create_statsig_feature_flag
    from intergrax.integrations.providers.workflow_orchestrator.prefect.bundle import create_prefect_workflow_orchestrator

    obs = create_newrelic_observability_backend(client=_FakeObsClient())
    assert_observability_backend(obs)

    issue = create_zendesk_issue_tracker(client=_FakeIssueClient())
    assert_issue_tracker(issue)

    flag = create_statsig_feature_flag(client=_FakeFlagClient())
    assert_feature_flag_backend(flag)

    workflow = create_prefect_workflow_orchestrator(client=_FakeWorkflowClient())
    assert_workflow_orchestrator_backend(workflow)


def test_p7_vision_ml_billing_crm_secrets() -> None:
    from intergrax.integrations.providers.vision_serving.triton.bundle import create_triton_vision_serving
    from intergrax.integrations.providers.ml_inference_host.replicate.bundle import create_replicate_ml_inference_host
    from intergrax.integrations.providers.billing_meter.stripe.bundle import create_stripe_billing_meter
    from intergrax.integrations.providers.crm.salesforce.bundle import create_salesforce_crm
    from intergrax.integrations.providers.secrets_store.infisical.bundle import create_infisical_secrets_store

    vision = create_triton_vision_serving(client=_FakeVisionClient())
    assert_vision_serving_backend(vision)

    ml = create_replicate_ml_inference_host(client=_FakeMlClient())
    assert_ml_inference_host_backend(ml)

    billing = create_stripe_billing_meter(client=_FakeBillingClient())
    assert_billing_meter_backend(billing)

    crm = create_salesforce_crm(client=_FakeCrmClient())
    assert_crm_backend(crm)

    secrets = create_infisical_secrets_store(client=_FakeSecretsClient())
    assert_secrets_store(secrets)


def test_harness_p6_presets_bind_categories() -> None:
    register_default_integrations()
    security = presets.harness_security_stack()
    assert security.slug_for_category(IntegrationCategory.SECURITY_SCANNER.value) == "trivy"
    assert security.slug_for_category(IntegrationCategory.SECRETS_STORE.value) == "infisical"

    sandbox = presets.harness_sandbox_stack()
    assert sandbox.slug_for_category(IntegrationCategory.SANDBOX_HOST.value) == "e2b"

    identity = presets.harness_identity_stack()
    assert identity.slug_for_category(IntegrationCategory.IDENTITY_PROVIDER.value) == "keycloak"

    gitops = presets.harness_gitops_stack()
    assert gitops.slug_for_category(IntegrationCategory.CI_CD.value) == "argocd"


def test_harness_m6_p6_probe_slugs_subset_of_catalog() -> None:
    register_default_integrations()
    catalog = catalog_snapshot()
    for slug in HARNESS_M6_P6_PROBE_SLUGS:
        assert slug in catalog


def test_harness_m6_p6_health_with_injected_clients() -> None:
    register_default_integrations()
    results = health_check_harness_m6_p6_probes()
    assert len(results) == len(HARNESS_M6_P6_PROBE_SLUGS)
    assert all(isinstance(item, HealthStatus) for item in results)


def test_integration_profile_harness_security_preset_roundtrip() -> None:
    register_default_integrations()
    profile = presets.harness_security_stack()
    restored = IntegrationProfile.model_validate(profile.model_dump(mode="json"))
    assert restored.slug_for_category(IntegrationCategory.SECURITY_SCANNER.value) == "trivy"
