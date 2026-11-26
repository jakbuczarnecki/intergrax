"""Red-first NT-REL probes (prefix-capable; assertion names FRZ reliability mechanisms)."""

from __future__ import annotations

import inspect
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


def test_nt_rel_09_kafka_rabbitmq_must_not_be_silently_rewritten_to_eager_celery() -> None:
    """NT-REL-09 / FRZ-REL-09: Kafka/RabbitMQ queue backends must not rewrite to eager Celery."""
    text = (_REPO / "intergrax/applications/_shared/queue_worker_wiring.py").read_text(encoding="utf-8")
    assert "fall back to eager Celery" not in text, (
        "NT-REL-09 queue/backpressure explicit: Kafka/RabbitMQ queue backends "
        "must not be silently rewritten to in-process eager Celery"
    )
    rewrite_lines = [
        line
        for line in text.splitlines()
        if "ProductionQueueBackend.CELERY" in line and "backend" in line.replace(" ", "")
    ]
    if rewrite_lines and "UnsupportedQueueBackendError" not in text:
        raise AssertionError(
            "NT-REL-09 queue/backpressure explicit: broker backends must not assign "
            "ProductionQueueBackend.CELERY without raising UnsupportedQueueBackendError"
        )


def test_nt_rel_10_in_memory_kubernetes_scaler_must_not_substitute_when_cluster_url_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NT-REL-10 / FRZ-REL-10: unset cluster URL must not silently use in-memory Kubernetes scaler."""
    monkeypatch.delenv("INTERGRAX_KUBERNETES_URL", raising=False)
    from intergrax.runtime.capacity.production_adapters import resolve_kubernetes_backend

    if "posture" in inspect.signature(resolve_kubernetes_backend).parameters:
        from intergrax.runtime.capacity.production_adapters import (
            CapacityBackendNotConfiguredError,
            CapacityBackendPosture,
        )

        with pytest.raises(CapacityBackendNotConfiguredError, match="in-memory scaler"):
            resolve_kubernetes_backend(posture=CapacityBackendPosture.PRODUCTION)
        return

    _backend, kind = resolve_kubernetes_backend()
    assert kind != "in_memory", (
        "NT-REL-10 resource saturation fail-closed: in-memory Kubernetes scaler "
        "must not substitute when INTERGRAX_KUBERNETES_URL is unset"
    )
    assert _backend is not None or os.environ.get("INTERGRAX_KUBERNETES_URL", "").strip()


def test_nt_rel_07_retry_engine_must_not_request_retry_past_declared_max_retries() -> None:
    """NT-REL-07 / FRZ-REL-07: no hidden infinite retry/fallback."""
    from intergrax.contracts.agent_contract_meta import AgentContract
    from intergrax.contracts.agent_run import AgentRunRequest, AgentRunResult
    from intergrax.contracts.validation import ValidationResult
    from intergrax.runtime.nexus.retry.retry_engine import RetryEngine, RetryPolicy
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.runtime.task.task import Task, TaskContext

    class _Rel07Agent:
        async def run(self, request: AgentRunRequest) -> AgentRunResult:
            _ = request
            raise NotImplementedError("proof-only")

        def get_contract(self) -> AgentContract:
            return AgentContract(
                id="rel07_agent",
                name="rel07",
                description="retry probe agent",
                capabilities=["rel07.cap"],
            )

    registry = AgentRegistry()
    registry.register(_Rel07Agent())
    engine = RetryEngine(registry, policy=RetryPolicy(max_retries=2))
    task = Task(
        tenant_id="t-rel07",
        user_id="u-rel07",
        message="retry probe",
        context=TaskContext(capability="rel07.cap"),
    )
    invalid = ValidationResult(valid=False, errors=["injected validation failure"])

    retry_past_budget = 0
    for attempt in range(5):
        decision = engine.decide(
            task,
            agent_id="rel07_agent",
            validation=invalid,
            attempt=attempt,
        )
        if attempt >= 2 and decision.should_retry:
            retry_past_budget += 1

    assert retry_past_budget == 0, (
        "NT-REL-07 no hidden infinite retry/fallback: "
        "RetryEngine.decide must return should_retry=False once attempt reaches declared max_retries"
    )
