# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Phase M.6 P4 catalog adapters."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Sequence

from intergrax.integrations.contracts.base import HealthStatus, IntegrationConfigurationError
from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe
from intergrax.integrations.contracts.ci_cd import CheckSuiteRecord, CiCdBackend, WorkflowRunRecord
from intergrax.integrations.contracts.cloud_platform import CloudPlatform
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend, FeatureFlagEvaluation
from intergrax.integrations.contracts.graph_store import GraphNodeRecord, GraphQueryResult, GraphStore
from intergrax.integrations.contracts.observability_backend import (
    MetricPoint,
    MetricQueryResult,
    MetricSeries,
    TraceQueryResult,
    TraceRecord,
)
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations._shared.p3.clients import Neo4jGraphStore, VaultSecretsStore
from intergrax.runtime.interactions.adapter_contract import InteractionAdapter
from intergrax.runtime.interactions.models import InboundInteraction


def _probe_client_health(client: object, *, slug: str, default_detail: str = "") -> HealthStatus:
    if isinstance(client, IntegrationHealthProbe):
        try:
            result = client.health()
        except Exception as exc:  # noqa: BLE001 — health probe surface
            return HealthStatus(slug=slug, healthy=False, detail=str(exc))
        if isinstance(result, HealthStatus):
            return result
        return HealthStatus(slug=slug, healthy=bool(result), detail=default_detail or "client probe")
    return HealthStatus(slug=slug, healthy=True, detail=default_detail or "no probe")


class RestSecretsStore:
    """HTTP secrets facade used by Doppler and similar providers."""

    def __init__(self, client: Any) -> None:
        self._client = client
        self._closed = False

    def get_secret(self, path: str, *, version: Optional[str] = None) -> str:
        self._require_open()
        return str(self._client.read_secret(path, version=version))

    def put_secret(self, path: str, value: str) -> None:
        self._require_open()
        self._client.write_secret(path, value)

    def delete_secret(self, path: str) -> None:
        self._require_open()
        self._client.delete_secret(path)

    def close(self) -> None:
        self._closed = True

    def health(self) -> HealthStatus | bool:
        if self._closed:
            return False
        if isinstance(self._client, IntegrationHealthProbe):
            try:
                result = self._client.health()
            except Exception:  # noqa: BLE001 — health probe surface
                return False
            if isinstance(result, HealthStatus):
                return result
            return bool(result)
        return True

    def _require_open(self) -> None:
        if self._closed:
            raise IntegrationConfigurationError("Secrets store is closed")


class CloudSecretsStore(VaultSecretsStore):
    """Cloud vendor secrets manager using mount-style paths (AWS/Azure/GCP)."""

    pass


class HttpFeatureFlagBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def is_enabled(self, flag_key: str, *, tenant_id: str, user_id: str = "") -> bool:
        return self.evaluate(flag_key, tenant_id=tenant_id, user_id=user_id).enabled

    def evaluate(
        self,
        flag_key: str,
        *,
        tenant_id: str,
        user_id: str = "",
    ) -> FeatureFlagEvaluation:
        payload = self._client.evaluate_flag(flag_key, tenant_id=tenant_id, user_id=user_id)
        if isinstance(payload, FeatureFlagEvaluation):
            return payload
        if isinstance(payload, dict):
            return FeatureFlagEvaluation(
                key=flag_key,
                enabled=bool(payload.get("enabled")),
                variant=str(payload.get("variant") or ""),
                metadata={k: str(v) for k, v in dict(payload.get("metadata") or {}).items()},
            )
        return FeatureFlagEvaluation(key=flag_key, enabled=bool(payload))

    def health(self) -> HealthStatus:
        return _probe_client_health(self._client, slug=self._provider)


class HttpCiCdBackend:
    def __init__(self, client: Any, *, provider: str) -> None:
        self._client = client
        self._provider = provider

    def get_workflow_run(self, run_id: str) -> WorkflowRunRecord:
        payload = self._client.get_workflow_run(run_id)
        if isinstance(payload, WorkflowRunRecord):
            return payload
        data = dict(payload or {})
        return WorkflowRunRecord(
            id=str(data.get("id") or run_id),
            name=str(data.get("name") or ""),
            status=str(data.get("status") or ""),
            conclusion=str(data.get("conclusion") or ""),
            url=str(data.get("url") or data.get("html_url") or ""),
        )

    def list_check_suites(self, *, ref: str, limit: int = 20) -> Sequence[CheckSuiteRecord]:
        rows = self._client.list_check_suites(ref=ref, limit=limit)
        suites: list[CheckSuiteRecord] = []
        for item in list(rows or [])[:limit]:
            if isinstance(item, CheckSuiteRecord):
                suites.append(item)
                continue
            data = dict(item or {})
            suites.append(
                CheckSuiteRecord(
                    id=str(data.get("id") or ""),
                    name=str(data.get("name") or ""),
                    status=str(data.get("status") or ""),
                    conclusion=str(data.get("conclusion") or ""),
                    url=str(data.get("url") or data.get("html_url") or ""),
                )
            )
        return suites

    def health(self) -> HealthStatus:
        return _probe_client_health(self._client, slug=self._provider)


class HttpObservabilityClientAdapter:
    """Wraps duck-typed HTTP metric/trace clients into ObservabilityBackend shape."""

    def __init__(
        self,
        client: Any,
        *,
        provider: str,
        instant_fn: Optional[Callable[[str, Optional[float]], float]] = None,
        range_fn: Optional[Callable[[str, float, float, str], list[dict[str, float]]]] = None,
        traces_fn: Optional[Callable[[int, Optional[str]], TraceQueryResult]] = None,
    ) -> None:
        self._client = client
        self._provider = provider
        self._instant_fn = instant_fn
        self._range_fn = range_fn
        self._traces_fn = traces_fn

    def query_instant(self, promql: str, *, eval_time: Optional[float] = None) -> MetricQueryResult:
        if self._instant_fn is not None:
            value = float(self._instant_fn(promql, eval_time))
        else:
            value = float(self._client.query_instant(promql, eval_time=eval_time))
        ts = float(eval_time or 0.0)
        return MetricQueryResult(
            result_type="vector",
            series=[MetricSeries(metric={"provider": self._provider}, points=[MetricPoint(timestamp=ts, value=value)])],
        )

    def query_range(
        self,
        promql: str,
        *,
        start: float,
        end: float,
        step: str = "15s",
    ) -> MetricQueryResult:
        if self._range_fn is not None:
            rows = self._range_fn(promql, start, end, step)
        else:
            rows = self._client.query_range(promql, start=start, end=end, step=step)
        points = [MetricPoint(timestamp=float(r["timestamp"]), value=float(r["value"])) for r in rows]
        return MetricQueryResult(
            result_type="matrix",
            series=[MetricSeries(metric={"provider": self._provider}, points=points)],
        )

    def query_traces(self, *, limit: int = 20, name: Optional[str] = None) -> TraceQueryResult:
        if self._traces_fn is not None:
            return self._traces_fn(limit, name)
        if hasattr(self._client, "query_traces"):
            result = self._client.query_traces(limit=limit, name=name)
            if isinstance(result, TraceQueryResult):
                return result
        if hasattr(self._client, "query_trace_by_id"):
            trace_id = name or ""
            if trace_id:
                payload = self._client.query_trace_by_id(trace_id)
                if payload:
                    return TraceQueryResult(
                        traces=[
                            TraceRecord(
                                trace_id=str(payload.get("traceID") or trace_id),
                                name=str(payload.get("name") or ""),
                                metadata={k: str(v) for k, v in dict(payload).items()},
                            )
                        ]
                    )
        return TraceQueryResult()

    def health(self) -> HealthStatus:
        return _probe_client_health(self._client, slug=self._provider)


class MemgraphGraphStore(Neo4jGraphStore):
    """Memgraph uses Bolt protocol compatible with Neo4j driver facade."""

    pass


class FalkorDbGraphStore(Neo4jGraphStore):
    """FalkorDB graph queries via Redis-module client facade."""

    pass


class KubernetesCloudPlatform:
    def __init__(self, client: Any, *, namespace: str, slug: str = "kubernetes") -> None:
        self._client = client
        self._namespace = namespace
        self._slug = slug

    @property
    def slug(self) -> str:
        return self._slug

    @property
    def default_region(self) -> Optional[str]:
        return self._namespace or None

    def resolve(self, category: str) -> Optional[str]:
        defaults: dict[str, str] = {
            "observability_backend": "prometheus",
            "secrets_store": "vault",
            "message_bus": "kafka",
            "object_storage": "s3",
        }
        return defaults.get(category.strip().lower())

    def health(self) -> HealthStatus:
        healthy = bool(self._client.health())
        return HealthStatus(slug=self._slug, healthy=healthy, detail=f"namespace={self._namespace}")


class MailgunInteractionAdapter(InteractionAdapter):
    def __init__(self, *, signing_key: str = "") -> None:
        self._signing_key = signing_key

    @property
    def channel(self) -> str:
        return "mailgun"

    def can_handle(self, payload: Mapping[str, Any]) -> bool:
        return "sender" in payload and ("body-plain" in payload or "stripped-text" in payload)

    def to_inbound(self, payload: Mapping[str, Any], *, tenant_id: str, user_id: str) -> InboundInteraction:
        message = str(payload.get("stripped-text") or payload.get("body-plain") or "")
        sender = str(payload.get("sender") or user_id)
        return InboundInteraction(
            tenant_id=tenant_id,
            user_id=sender,
            session_id=str(payload.get("Message-Id") or payload.get("message-id") or ""),
            message=message,
            channel=self.channel,
            interaction_id=str(payload.get("Message-Id") or ""),
            metadata={"subject": str(payload.get("subject") or "")},
        )


class OllamaInteractionAdapter(InteractionAdapter):
    """Health/model-list surface for local Ollama host (modality bridge, not LLM catalog)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @property
    def channel(self) -> str:
        return "ollama"

    def can_handle(self, payload: Mapping[str, Any]) -> bool:
        return payload.get("kind") == "ollama" or "model" in payload

    def to_inbound(self, payload: Mapping[str, Any], *, tenant_id: str, user_id: str) -> InboundInteraction:
        model = str(payload.get("model") or "")
        prompt = str(payload.get("prompt") or payload.get("message") or "")
        return InboundInteraction(
            tenant_id=tenant_id,
            user_id=user_id,
            session_id=str(payload.get("session_id") or ""),
            message=prompt,
            channel=self.channel,
            metadata={"model": model, "models": json.dumps(self._client.list_models())},
        )

    def list_models(self) -> list[str]:
        return list(self._client.list_models())

    def health(self) -> HealthStatus:
        healthy = bool(self._client.health())
        return HealthStatus(slug="ollama", healthy=healthy, detail="local inference host")


__all__ = [
    "CloudSecretsStore",
    "FalkorDbGraphStore",
    "HttpCiCdBackend",
    "HttpFeatureFlagBackend",
    "HttpObservabilityClientAdapter",
    "KubernetesCloudPlatform",
    "MailgunInteractionAdapter",
    "MemgraphGraphStore",
    "OllamaInteractionAdapter",
    "RestSecretsStore",
]
