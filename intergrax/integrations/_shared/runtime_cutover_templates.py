# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime cutover code templates for INTEGRATIONS-2E provider migration."""

from __future__ import annotations

# Category → runtime protocol delegation specs for generated integration.py bodies.
CATEGORY_RUNTIME_SPECS: dict[str, dict[str, object]] = {
    "vector_store": {
        "protocol_import": "from intergrax.integrations.contracts.vector_store import MetadataFilter, VectorStore, VectorStoreHit",
        "protocol_name": "VectorStore",
        "runtime_attr": "_inner",
        "from_method": "from_store",
        "runtime_param": "inner",
        "config_param": "store_config",
        "config_type_suffix": "IntegrationConfig",
        "register_protocol": True,
        "extra_imports": "from intergrax.integrations.contracts.health_probe import IntegrationHealthProbe",
        "extra_properties": (
            '    @property\n'
            '    def rag_store(self) -> VectorStore:\n'
            '        return self._require_runtime()\n'
        ),
        "methods": (
            "    def add_documents(\n"
            "        self,\n"
            "        documents: Sequence[Any],\n"
            "        embeddings: Sequence[Sequence[float]],\n"
            "        *,\n"
            "        ids: Sequence[str] | None = None,\n"
            "    ) -> None:\n"
            "        self._require_runtime().add_documents(documents, embeddings, ids=ids)\n\n"
            "    def query(\n"
            "        self,\n"
            "        query_embedding: Sequence[float],\n"
            "        *,\n"
            "        top_k: int,\n"
            "        metadata_filter: MetadataFilter | None = None,\n"
            "        include_embeddings: bool = False,\n"
            "    ) -> list[VectorStoreHit]:\n"
            "        return self._require_runtime().query(\n"
            "            query_embedding,\n"
            "            top_k=top_k,\n"
            "            metadata_filter=metadata_filter,\n"
            "            include_embeddings=include_embeddings,\n"
            "        )\n\n"
            "    def delete(self, ids: Sequence[str]) -> None:\n"
            "        self._require_runtime().delete(ids)\n\n"
            "    def count(self) -> int:\n"
            "        return self._require_runtime().count()\n"
        ),
    },
    "object_storage": {
        "protocol_import": "from intergrax.integrations.contracts.object_storage import ObjectStorage, PresignedUrlMethod, StoredObject",
        "protocol_name": "ObjectStorage",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": True,
        "methods": (
            "    def put(\n"
            "        self,\n"
            "        key: str,\n"
            "        body: bytes,\n"
            "        *,\n"
            "        content_type: str = \"application/octet-stream\",\n"
            "        metadata: Mapping[str, str] | None = None,\n"
            "    ) -> None:\n"
            "        self._require_runtime().put(key, body, content_type=content_type, metadata=metadata)\n\n"
            "    def get(self, key: str) -> StoredObject | None:\n"
            "        return self._require_runtime().get(key)\n\n"
            "    def delete(self, key: str) -> None:\n"
            "        self._require_runtime().delete(key)\n\n"
            "    def presigned_url(\n"
            "        self,\n"
            "        key: str,\n"
            "        *,\n"
            "        expires_in_seconds: int = 3600,\n"
            "        method: PresignedUrlMethod = \"GET\",\n"
            "    ) -> str:\n"
            "        return self._require_runtime().presigned_url(\n"
            "            key,\n"
            "            expires_in_seconds=expires_in_seconds,\n"
            "            method=method,\n"
            "        )\n\n"
            "    def close(self) -> None:\n"
            "        self._require_runtime().close()\n"
        ),
        "typing_imports": "Mapping, ",
    },
    "search_provider": {
        "protocol_import": "from intergrax.integrations.contracts.search_provider import SearchProvider",
        "protocol_name": "SearchProvider",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": True,
        "methods": (
            "    def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:\n"
            "        return self._require_runtime().search(query, limit=limit)\n"
        ),
    },
    "observability_backend": {
        "protocol_import": (
            "from intergrax.integrations.contracts.observability_backend import "
            "MetricQueryResult, ObservabilityBackend, TraceQueryResult"
        ),
        "protocol_name": "ObservabilityBackend",
        "runtime_attr": "_backend",
        "from_method": "from_backend",
        "runtime_param": "backend",
        "register_protocol": True,
        "uses_transport": True,
        "methods": (
            "    def query_instant(self, promql: str, *, eval_time: float | None = None) -> MetricQueryResult:\n"
            "        return self._require_runtime().query_instant(promql, eval_time=eval_time)\n\n"
            "    def query_range(\n"
            "        self,\n"
            "        promql: str,\n"
            "        *,\n"
            "        start: float,\n"
            "        end: float,\n"
            "        step: str = \"15s\",\n"
            "    ) -> MetricQueryResult:\n"
            "        return self._require_runtime().query_range(\n"
            "            promql,\n"
            "            start=start,\n"
            "            end=end,\n"
            "            step=step,\n"
            "        )\n\n"
            "    def query_traces(\n"
            "        self,\n"
            "        *,\n"
            "        limit: int = 20,\n"
            "        name: str | None = None,\n"
            "    ) -> TraceQueryResult:\n"
            "        return self._require_runtime().query_traces(limit=limit, name=name)\n"
        ),
    },
    "relational_store": {
        "protocol_import": "from intergrax.integrations.contracts.relational_store import RelationalStore",
        "protocol_name": "RelationalStore",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": True,
        "methods": (
            "    def connect(self) -> None:\n"
            "        self._require_runtime().connect()\n\n"
            "    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:\n"
            "        self._require_runtime().execute(sql, params)\n\n"
            "    def fetch_all(self, sql: str, params: Sequence[Any] = ()) -> Sequence[Mapping[str, Any]]:\n"
            "        return self._require_runtime().fetch_all(sql, params)\n\n"
            "    def close(self) -> None:\n"
            "        self._require_runtime().close()\n"
        ),
        "typing_imports": "Mapping, ",
    },
    "notification_channel": {
        "protocol_import": "from intergrax.integrations.contracts.notification_channel import NotificationChannel",
        "protocol_name": "NotificationChannel",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": False,
        "methods": (
            "    async def notify(self, message: Any) -> None:\n"
            "        await self._require_runtime().notify(message)\n\n"
            "    def health(self) -> Any:\n"
            "        return self._require_runtime().health()\n"
        ),
    },
    "message_bus": {
        "protocol_import": "from intergrax.integrations.contracts.message_bus import MessageBus",
        "protocol_name": "MessageBus",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": False,
        "methods": (
            "    def publish(self, topic: str, payload: bytes, *, headers: Mapping[str, str] | None = None) -> None:\n"
            "        self._require_runtime().publish(topic, payload, headers=headers)\n\n"
            "    def close(self) -> None:\n"
            "        self._require_runtime().close()\n"
        ),
        "typing_imports": "Mapping, ",
    },
    "issue_tracker": {
        "protocol_import": "from intergrax.integrations.contracts.issue_tracker import IssueRecord, IssueSearchResult, IssueTracker",
        "protocol_name": "IssueTracker",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": False,
        "methods": (
            "    def search_issues(self, query: str, *, limit: int = 20) -> IssueSearchResult:\n"
            "        return self._require_runtime().search_issues(query, limit=limit)\n\n"
            "    def get_issue(self, issue_id: str) -> IssueRecord | None:\n"
            "        return self._require_runtime().get_issue(issue_id)\n\n"
            "    def create_issue(self, *, title: str, body: str = \"\", labels: Sequence[str] = ()) -> IssueRecord:\n"
            "        return self._require_runtime().create_issue(title=title, body=body, labels=labels)\n"
        ),
    },
    "browser_automation": {
        "protocol_import": "from intergrax.integrations.contracts.browser_automation import BrowserAutomation, PageContent",
        "protocol_name": "BrowserAutomation",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": False,
        "methods": (
            "    def fetch_page(self, url: str, *, wait_until: str = \"load\") -> PageContent:\n"
            "        return self._require_runtime().fetch_page(url, wait_until=wait_until)\n\n"
            "    def close(self) -> None:\n"
            "        self._require_runtime().close()\n"
        ),
    },
    "secrets_store": {
        "protocol_import": "from intergrax.integrations.contracts.secrets_store import SecretsStore",
        "protocol_name": "SecretsStore",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": False,
        "methods": (
            "    def get_secret(self, key: str) -> str | None:\n"
            "        return self._require_runtime().get_secret(key)\n\n"
            "    def set_secret(self, key: str, value: str) -> None:\n"
            "        self._require_runtime().set_secret(key, value)\n\n"
            "    def delete_secret(self, key: str) -> None:\n"
            "        self._require_runtime().delete_secret(key)\n"
        ),
    },
    "graph_store": {
        "protocol_import": "from intergrax.integrations.contracts.graph_store import GraphQueryResult, GraphStore",
        "protocol_name": "GraphStore",
        "runtime_attr": "_runtime",
        "from_method": "from_runtime",
        "runtime_param": "runtime",
        "register_protocol": False,
        "methods": (
            "    def query(self, query: str, *, params: Mapping[str, Any] | None = None) -> GraphQueryResult:\n"
            "        return self._require_runtime().query(query, params=params)\n\n"
            "    def close(self) -> None:\n"
            "        self._require_runtime().close()\n"
        ),
        "typing_imports": "Mapping, ",
    },
}

# Categories without explicit runtime delegation use generic from_runtime + __getattr__ shim.
GENERIC_RUNTIME_CATEGORIES: frozenset[str] = frozenset(
    {
        "document_store",
        "key_value_cache",
        "interaction_surface",
        "collaboration_suite",
        "wiki_knowledge",
        "cloud_platform",
        "document_parser",
        "rerank_provider",
        "feature_flag",
        "ci_cd",
        "security_scanner",
        "sandbox_host",
        "identity_provider",
        "speech_provider",
        "workflow_orchestrator",
        "billing_meter",
        "crm",
        "vision_serving",
        "ml_inference_host",
    }
)
