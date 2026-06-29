#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Inline legacy runtime delegation into provider integration.py (non-observability wave)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS_ROOT = ROOT / "intergrax" / "integrations" / "providers"

SKIP_CATEGORIES = frozenset({"observability_backend", "llm_guardrail"})

DEFERRED_SLUGS = frozenset(
    {
        "llm_guard",
        "guardrails_ai",
        "nemo_guardrails",
        "openguardrails",
        "presidio",
        "llama_guard",
        "lakera",
        "azure_content_safety",
        "bedrock_guardrails",
        "qdrant",
        "pinecone",
    }
)

CATEGORY_PROTOCOL: dict[str, tuple[str, str]] = {
    "relational_store": ("intergrax.integrations.contracts.relational_store", "RelationalStore"),
    "document_store": ("intergrax.integrations.contracts.document_store", "DocumentStore"),
    "key_value_cache": ("intergrax.integrations.contracts.key_value_cache", "KeyValueCache"),
    "graph_store": ("intergrax.integrations.contracts.graph_store", "GraphStore"),
    "message_bus": ("intergrax.integrations.contracts.message_bus", "MessageBus"),
    "notification_channel": (
        "intergrax.integrations.contracts.notification_channel",
        "NotificationChannel",
    ),
    "object_storage": ("intergrax.integrations.contracts.object_storage", "ObjectStorage"),
    "vector_store": ("intergrax.integrations.contracts.vector_store", "VectorStore"),
    "search_provider": ("intergrax.integrations.contracts.search_provider", "SearchProvider"),
    "rerank_provider": ("intergrax.integrations.contracts.rerank_provider", "RerankProvider"),
    "interaction_surface": (
        "intergrax.integrations.contracts.interaction_surface",
        "InteractionSurface",
    ),
    "collaboration_suite": (
        "intergrax.integrations.contracts.collaboration_suite",
        "CollaborationSuite",
    ),
    "issue_tracker": ("intergrax.integrations.contracts.issue_tracker", "IssueTracker"),
    "wiki_knowledge": ("intergrax.integrations.contracts.wiki_knowledge", "WikiKnowledge"),
    "browser_automation": (
        "intergrax.integrations.contracts.browser_automation",
        "BrowserAutomation",
    ),
    "billing_meter": ("intergrax.integrations.contracts.billing_meter", "BillingMeterBackend"),
    "crm": ("intergrax.integrations.contracts.crm", "CrmBackend"),
    "cloud_platform": ("intergrax.integrations.contracts.cloud_platform", "CloudPlatform"),
    "ci_cd": ("intergrax.integrations.contracts.ci_cd", "CiCdBackend"),
    "security_scanner": (
        "intergrax.integrations.contracts.security_scanner",
        "SecurityScannerBackend",
    ),
    "sandbox_host": ("intergrax.integrations.contracts.sandbox_host", "SandboxHostBackend"),
    "workflow_orchestrator": (
        "intergrax.integrations.contracts.workflow_orchestrator",
        "WorkflowOrchestratorBackend",
    ),
    "secrets_store": ("intergrax.integrations.contracts.secrets_store", "SecretsStore"),
    "feature_flag": ("intergrax.integrations.contracts.feature_flag", "FeatureFlagBackend"),
    "identity_provider": (
        "intergrax.integrations.contracts.identity_provider",
        "IdentityProviderBackend",
    ),
    "speech_provider": (
        "intergrax.integrations.contracts.speech_provider",
        "SpeechProviderBackend",
    ),
    "vision_serving": (
        "intergrax.integrations.contracts.vision_serving",
        "VisionServingBackend",
    ),
    "ml_inference_host": (
        "intergrax.integrations.contracts.ml_inference_host",
        "MlInferenceHostBackend",
    ),
    "document_parser": ("intergrax.integrations.contracts.document_parser", "DocumentParser"),
}

REQUIRE_RUNTIME_BLOCK = re.compile(
    r"\n    def _require_runtime\(self\) -> Any:.*?        return runtime\n",
    re.MULTILINE | re.DOTALL,
)

REQUIRE_RUNTIME_SIMPLE = re.compile(
    r"\n    def _require_runtime\(self\) -> Any:\n"
    r"        if self\._runtime is None:\n"
    r"            raise IntegrationConfigurationError\([^\)]+\)\n"
    r"        return self\._runtime\n",
    re.MULTILINE,
)

GETATTR_BLOCK = re.compile(
    r"\n    def __getattr__\(self, name: str\) -> object:.*?        return getattr\(self\._require_(?:runtime|client)\(\), name\)\n",
    re.MULTILINE | re.DOTALL,
)

FROM_RUNTIME_BLOCK = re.compile(
    r"\n    @classmethod\n"
    r"    def from_runtime\(\n"
    r"        cls,\n"
    r"        runtime: Any,\n"
    r"        \*,\n"
    r"        enabled: bool = True,\n"
    r"    \) -> [^\n]+:\n"
    r"        integration = cls\.for_provider\(\n"
    r"            provider_id=[^\n]+,\n"
    r"            display_name=[^\n]+,\n"
    r"            config=[^\n]+\(enabled=enabled\),\n"
    r"        \)\n"
    r"        integration\._runtime = runtime\n"
    r"        return integration\n",
    re.MULTILINE,
)

FROM_RUNTIME_ONELINE = re.compile(
    r"\n    @classmethod\n"
    r"    def from_runtime\(cls, runtime: Any, \*, enabled: bool = True\) -> [^\n]+:\n"
    r"        integration = cls\.for_provider\(\n"
    r"            provider_id=[^\n]+,\n"
    r"            display_name=[^\n]+,\n"
    r"            config=[^\n]+\(enabled=enabled\),\n"
    r"        \)\n"
    r"        integration\._runtime = runtime\n"
    r"        return integration\n",
    re.MULTILINE,
)

FROM_RUNTIME_VECTOR = re.compile(
    r"\n    @classmethod\n"
    r"    def from_runtime\(\n"
    r"        cls,\n"
    r"        runtime: Any,\n"
    r"        \*,\n"
    r"        enabled: bool = True,\n"
    r"        store_config: Any \| None = None,\n"
    r"    \) -> [^\n]+:\n"
    r"        return cls\.from_store\([^\n]+\)\n",
    re.MULTILINE,
)

STUB_CLIENT = re.compile(
    r"@runtime_checkable\nclass \w+Client\(Protocol\):\n"
    r'    """Injectable client facade — no vendor SDK or network I/O in the integration class\."""\n\n'
    r"    async def ping\(self\) -> None:\n"
    r'        """Lightweight connectivity check\."""\n\n\n',
    re.MULTILINE,
)


def _require_client_block(protocol: str) -> str:
    return f"""
    def _require_client(self) -> {protocol}:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{{type(self).__name__}} requires a catalog client for operations",
            )
        return self._client
"""


def _require_inner_block() -> str:
    return """
    def _require_inner(self) -> VectorStore:
        if self._inner is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires an inner store for vector operations",
            )
        return self._inner
"""


def _slug_prefix(slug: str, category: str) -> str:
    overrides = {
        "aws": "Aws",
        "gcp": "Gcp",
        "mssql": "Mssql",
        "pgvector": "Pgvector",
        "yt_dlp": "YtDlp",
        "e2b": "E2b",
        "n8n": "N8n",
        "okta": "Okta",
        "auth0": "Auth0",
    }
    if slug in overrides:
        base = overrides[slug]
    else:
        base = "".join(part.capitalize() for part in slug.split("_"))
    category_parts = "".join(part.capitalize() for part in category.split("_"))
    return f"{base}{category_parts}"


def _ensure_protocol_import(src: str, module: str, protocol: str) -> str:
    import_line = f"from {module} import {protocol}\n"
    if import_line in src:
        return src
    marker = "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
    if marker in src:
        return src.replace(marker, marker + import_line)
    return src


def _replace_stub_client(src: str, prefix: str, protocol: str) -> str:
    alias = f"{prefix}Client = {protocol}\n\n"
    return STUB_CLIENT.sub(alias, src, count=1)


def migrate_vector_integration(src: str) -> str:
    src = src.replace("_require_runtime()", "_require_inner()")
    src = src.replace("_inner: Any | None = PrivateAttr(default=None)", "_inner: VectorStore | None = PrivateAttr(default=None)")
    src = src.replace("return self._require_inner()", "return self._require_inner()")
    src = src.replace(
        "    def rag_store(self) -> VectorStore:\n        return self._require_runtime()",
        "    def rag_store(self) -> VectorStore:\n        return self._require_inner()",
    )
    src = REQUIRE_RUNTIME_BLOCK.sub(_require_inner_block(), src, count=1)
    src = FROM_RUNTIME_VECTOR.sub("\n", src, count=1)
    src = src.replace(
        "Legacy catalog factory (create_", "Legacy catalog factories construct this class. Catalog factory (create_",
    )
    if "VectorStore | None" in src and "from intergrax.integrations.contracts.vector_store import" not in src:
        marker = "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
        src = src.replace(
            marker,
            marker + "from intergrax.integrations.contracts.vector_store import VectorStore\n",
        )
    return src


def migrate_standard_integration(category: str, slug: str, src: str) -> str:
    protocol_module, protocol = CATEGORY_PROTOCOL[category]
    prefix = _slug_prefix(slug, category)

    src = src.replace("_runtime: Any | None = PrivateAttr(default=None)\n", "")
    src = src.replace("_backend: Any | None = PrivateAttr(default=None)\n", "")
    src = src.replace("_require_runtime()", "_require_client()")
    src = src.replace(
        "requires a runtime delegate for catalog operations",
        "requires a catalog client for operations",
    )
    src = src.replace(
        "delegates to this class.",
        "owns catalog behavior; legacy factories use from_client().",
    )
    src = REQUIRE_RUNTIME_BLOCK.sub(_require_client_block(protocol), src, count=1)
    src = REQUIRE_RUNTIME_SIMPLE.sub(_require_client_block(protocol), src, count=1)
    src = GETATTR_BLOCK.sub("\n", src, count=1)
    src = FROM_RUNTIME_BLOCK.sub("\n", src, count=1)
    src = FROM_RUNTIME_ONELINE.sub("\n", src, count=1)

    src = _ensure_protocol_import(src, protocol_module, protocol)
    src = _replace_stub_client(src, prefix, protocol)

    if "from typing import Any, Protocol" in src and "Any" not in src.split("@runtime_checkable", 1)[0]:
        src = src.replace("from typing import Any, Protocol", "from typing import Protocol")
    elif "from typing import Any, Protocol, Sequence" in src:
        if "Any" not in re.sub(r"from typing import[^\n]+\n", "", src, count=1).split("class ", 1)[0]:
            src = src.replace("from typing import Any, Protocol, Sequence", "from typing import Protocol, Sequence")
    elif "from typing import Any, Protocol, Sequence, Mapping" in src:
        tail = src.split("class ", 1)[0]
        if "Any" not in tail.replace("from typing import Any, Protocol, Sequence, Mapping", ""):
            src = src.replace(
                "from typing import Any, Protocol, Sequence, Mapping",
                "from typing import Protocol, Sequence, Mapping",
            )

    if "runtime_checkable" in src and "@runtime_checkable" not in src:
        src = src.replace("from typing import Protocol", "from typing import Protocol, runtime_checkable")

    return src


def migrate_integration(category: str, slug: str) -> bool:
    path = PROVIDERS_ROOT / category / slug / "integration.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    if "def _require_runtime" not in src and "def __getattr__" not in src:
        return False
    if category == "vector_store":
        new_src = migrate_vector_integration(src)
    elif category not in CATEGORY_PROTOCOL:
        return False
    else:
        new_src = migrate_standard_integration(category, slug, src)
    if new_src == src:
        return False
    path.write_text(new_src, encoding="utf-8")
    return True


def migrate_bundle(category: str, slug: str) -> bool:
    path = PROVIDERS_ROOT / category / slug / "bundle.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    if ".from_runtime(" not in src:
        return False
    src = src.replace(".from_runtime(", ".from_client(")
    path.write_text(src, encoding="utf-8")
    return True


def migrate_opens(category: str, slug: str) -> bool:
    path = PROVIDERS_ROOT / category / slug / "opens.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    if ".from_runtime(" not in src:
        return False
    src = src.replace(".from_runtime(", ".from_client(")
    path.write_text(src, encoding="utf-8")
    return True


def main() -> None:
    integrations = 0
    bundles = 0
    opens = 0
    for category_dir in sorted(PROVIDERS_ROOT.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        category = category_dir.name
        if category in SKIP_CATEGORIES:
            continue
        for slug_dir in sorted(category_dir.iterdir()):
            if not slug_dir.is_dir() or slug_dir.name.startswith("_"):
                continue
            slug = slug_dir.name
            if slug in DEFERRED_SLUGS:
                continue
            if migrate_integration(category, slug):
                integrations += 1
            if migrate_bundle(category, slug):
                bundles += 1
            if migrate_opens(category, slug):
                opens += 1
    print(f"migrated integrations: {integrations}, bundles: {bundles}, opens: {opens}")


if __name__ == "__main__":
    main()
