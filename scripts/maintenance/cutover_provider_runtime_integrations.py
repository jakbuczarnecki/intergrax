#!/usr/bin/env python3
"""Apply INTEGRATIONS-2E runtime cutover to provider packages."""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from intergrax.integrations._shared.runtime_cutover_templates import (  # noqa: E402
    CATEGORY_RUNTIME_SPECS,
    GENERIC_RUNTIME_CATEGORIES,
)
from intergrax.integrations.providers.layout import SLUG_CATEGORY  # noqa: E402

H = "# © Artur Czarnecki. All rights reserved.\n# Intergrax framework – proprietary and confidential.\n\n"

DEFERRED_SLUGS: frozenset[str] = frozenset(
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
    }
)

ALREADY_CUTOVER: frozenset[str] = frozenset({"pinecone", "qdrant"})

_CLASS_NAME_OVERRIDES: dict[str, str] = {
    "newrelic": "NewRelic",
    "opentelemetry_collector": "OpenTelemetryCollector",
    "aws": "Aws",
    "gcp": "Gcp",
    "azure_sql": "AzureSql",
    "cloud_sql": "CloudSql",
    "mssql": "Mssql",
    "pgvector": "Pgvector",
    "yt_dlp": "YtDlp",
    "e2b": "E2b",
    "n8n": "N8n",
    "okta": "Okta",
    "auth0": "Auth0",
    "ci_cd": "CiCd",
}

_CATEGORY_CONTRACT: dict[str, tuple[str, str]] = {
    "relational_store": ("intergrax.runtime.integrations.categories.data", "RelationalStoreIntegrationContract"),
    "document_store": ("intergrax.runtime.integrations.categories.data", "DocumentStoreIntegrationContract"),
    "key_value_cache": ("intergrax.runtime.integrations.categories.data", "KeyValueCacheIntegrationContract"),
    "graph_store": ("intergrax.runtime.integrations.categories.data", "GraphStoreIntegrationContract"),
    "message_bus": ("intergrax.runtime.integrations.categories.messaging", "MessageBusIntegrationContract"),
    "notification_channel": (
        "intergrax.runtime.integrations.categories.messaging",
        "NotificationChannelIntegrationContract",
    ),
    "object_storage": ("intergrax.runtime.integrations.categories.storage", "ObjectStorageIntegrationContract"),
    "vector_store": ("intergrax.runtime.integrations.categories.storage", "VectorStoreIntegrationContract"),
    "search_provider": ("intergrax.runtime.integrations.categories.search", "SearchProviderIntegrationContract"),
    "rerank_provider": ("intergrax.runtime.integrations.categories.search", "RerankProviderIntegrationContract"),
    "interaction_surface": (
        "intergrax.runtime.integrations.categories.collaboration",
        "InteractionSurfaceIntegrationContract",
    ),
    "collaboration_suite": (
        "intergrax.runtime.integrations.categories.collaboration",
        "CollaborationSuiteIntegrationContract",
    ),
    "issue_tracker": ("intergrax.runtime.integrations.categories.collaboration", "IssueTrackerIntegrationContract"),
    "wiki_knowledge": ("intergrax.runtime.integrations.categories.collaboration", "WikiKnowledgeIntegrationContract"),
    "browser_automation": (
        "intergrax.runtime.integrations.categories.automation",
        "BrowserAutomationIntegrationContract",
    ),
    "billing_meter": ("intergrax.runtime.integrations.categories.automation", "BillingMeterIntegrationContract"),
    "crm": ("intergrax.runtime.integrations.categories.automation", "CrmIntegrationContract"),
    "cloud_platform": ("intergrax.runtime.integrations.categories.devops", "CloudPlatformIntegrationContract"),
    "ci_cd": ("intergrax.runtime.integrations.categories.devops", "CiCdIntegrationContract"),
    "security_scanner": ("intergrax.runtime.integrations.categories.devops", "SecurityScannerIntegrationContract"),
    "sandbox_host": ("intergrax.runtime.integrations.categories.devops", "SandboxHostIntegrationContract"),
    "workflow_orchestrator": (
        "intergrax.runtime.integrations.categories.devops",
        "WorkflowOrchestratorIntegrationContract",
    ),
    "secrets_store": ("intergrax.runtime.integrations.categories.security", "SecretsStoreIntegrationContract"),
    "feature_flag": ("intergrax.runtime.integrations.categories.security", "FeatureFlagIntegrationContract"),
    "identity_provider": (
        "intergrax.runtime.integrations.categories.security",
        "IdentityProviderIntegrationContract",
    ),
    "speech_provider": ("intergrax.runtime.integrations.categories.ai", "SpeechProviderIntegrationContract"),
    "vision_serving": ("intergrax.runtime.integrations.categories.ai", "VisionServingIntegrationContract"),
    "ml_inference_host": ("intergrax.runtime.integrations.categories.ai", "MlInferenceHostIntegrationContract"),
    "document_parser": ("intergrax.runtime.integrations.categories.ai", "DocumentParserIntegrationContract"),
    "observability_backend": (
        "intergrax.runtime.integrations.observability",
        "ObservabilityVendorIntegrationContract",
    ),
}


def slug_to_pascal(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def category_to_pascal(category: str) -> str:
    return "".join(part.capitalize() for part in category.split("_"))


def class_prefix(slug: str, category: str) -> str:
    if category == "observability_backend":
        return f"{slug_to_pascal(slug)}Observability"
    return f"{slug_to_pascal(slug)}{category_to_pascal(category)}"


def provider_id_const(slug: str, category: str) -> str:
    if category == "observability_backend":
        return f"{slug.upper()}_OBSERVABILITY_PROVIDER_ID"
    return f"{slug.upper()}_{category.upper()}_PROVIDER_ID"


def display_name(slug: str) -> str:
    return slug.replace("_", " ").title()


def pkg_path(slug: str, category: str) -> Path:
    return ROOT / "intergrax" / "integrations" / "providers" / category / slug


def is_cutover_done(path: Path) -> bool:
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    return "INTEGRATIONS-2E runtime cutover" in text


def detect_legacy_factory(slug: str, category: str, pkg: Path) -> str:
    bundle_path = pkg / "bundle.py"
    register_path = pkg / "register.py"
    candidates: list[str] = []
    contract_name = f"create_{slug}_{category}_integration"
    if bundle_path.is_file():
        src = bundle_path.read_text(encoding="utf-8")
        match = re.search(r"__all__\s*=\s*\[(.*?)\]", src, re.S)
        if match:
            candidates.extend(re.findall(r'"(create_[^"]+)"', match.group(1)))
        candidates.extend(re.findall(r"def (create_\w+)\(", src))
    if register_path.is_file():
        src = register_path.read_text(encoding="utf-8")
        for token in re.findall(r"(create_\w+)", src):
            if token != contract_name and token.startswith("create_"):
                candidates.append(token)
    legacy = [name for name in dict.fromkeys(candidates) if name != contract_name]
    if not legacy:
        raise RuntimeError(f"{slug}: no legacy factory")
    for name in legacy:
        if name.startswith(f"create_{slug}_"):
            return name
    return legacy[0]


def _parse_observability_constants(source: str) -> tuple[str | None, str | None]:
    signals_match = re.search(r"^([A-Z_]+SUPPORTED_SIGNALS)\s*=", source, re.M)
    transport_match = re.search(r"class (\w+Transport)\(", source)
    return (
        signals_match.group(1) if signals_match else None,
        transport_match.group(1) if transport_match else None,
    )


def _runtime_block(
    category: str,
    prefix: str,
    slug: str,
    const: str,
    label: str,
    *,
    signals_const: str | None = None,
) -> str:
    if category == "observability_backend":
        spec = CATEGORY_RUNTIME_SPECS["observability_backend"]
        signals_ref = signals_const or f"{slug.upper()}_SUPPORTED_SIGNALS"
        return (
            f"    {spec['runtime_attr']}: Any | None = PrivateAttr(default=None)\n\n"
            f"    @classmethod\n"
            f"    def from_backend(\n"
            f"        cls,\n"
            f"        backend: Any,\n"
            f"        *,\n"
            f"        enabled: bool = True,\n"
            f"        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,\n"
            f"    ) -> {prefix}Integration:\n"
            f"        signals = supported_signals or {signals_ref}\n"
            f"        integration = cls.for_provider(\n"
            f"            provider_id={const},\n"
            f"            supported_signals=signals,\n"
            f'            display_name="{label}",\n'
            f"            config={prefix}IntegrationConfig(enabled=enabled),\n"
            f"        )\n"
            f"        integration.{spec['runtime_attr']} = backend\n"
            f"        return integration\n\n"
            f"    @property\n"
            f"    def backend(self) -> Any | None:\n"
            f"        return self.{spec['runtime_attr']}\n\n"
            f"{spec['methods']}\n\n"
            f"    def _require_runtime(self) -> Any:\n"
            f"        runtime = self.{spec['runtime_attr']}\n"
            f"        if runtime is None:\n"
            f'            raise IntegrationConfigurationError(\n'
            f'                "{label} integration requires a runtime backend for catalog operations",\n'
            f"            )\n"
            f"        return runtime\n"
        )

    if category in CATEGORY_RUNTIME_SPECS:
        spec = CATEGORY_RUNTIME_SPECS[category]
        from_method = spec["from_method"]
        runtime_param = spec["runtime_param"]
        runtime_attr = spec["runtime_attr"]
        extra = spec.get("extra_properties", "")
        config_block = ""
        config_prop = ""
        store_config_attr = ""
        if category == "vector_store":
            store_config_attr = "    _store_config: Any | None = PrivateAttr(default=None)\n"
            config_block = "        store_config: Any | None = None,\n"
            config_prop = (
                "    @property\n"
                "    def store_config(self) -> Any | None:\n"
                "        return self._store_config\n\n"
            )
        from_runtime_block = ""
        if from_method != "from_runtime":
            from_runtime_sig = (
                "    @classmethod\n"
                "    def from_runtime(cls, runtime: Any, *, enabled: bool = True"
                + (", store_config: Any | None = None" if category == "vector_store" else "")
                + f") -> {prefix}Integration:\n"
            )
            from_runtime_body = (
                f"        return cls.{from_method}(runtime, enabled=enabled"
                + (", store_config=store_config" if category == "vector_store" else "")
                + ")\n"
            )
            from_runtime_block = from_runtime_sig + from_runtime_body + "\n"
        return (
            store_config_attr
            + f"    {runtime_attr}: Any | None = PrivateAttr(default=None)\n\n"
            + "    @classmethod\n"
            + f"    def {from_method}(\n"
            + "        cls,\n"
            + f"        {runtime_param}: Any,\n"
            + "        *,\n"
            + "        enabled: bool = True,\n"
            + config_block
            + f"    ) -> {prefix}Integration:\n"
            + "        integration = cls.for_provider(\n"
            + f"            provider_id={const},\n"
            + f'            display_name="{label}",\n'
            + f"            config={prefix}IntegrationConfig(enabled=enabled),\n"
            + "        )\n"
            + ("        integration._store_config = store_config\n" if category == "vector_store" else "")
            + f"        integration.{runtime_attr} = {runtime_param}\n"
            + "        return integration\n\n"
            + from_runtime_block
            + config_prop
            + str(extra)
            + str(spec["methods"])
            + "\n\n"
            + "    def _require_runtime(self) -> Any:\n"
            + f"        runtime = self.{runtime_attr}\n"
            + "        if runtime is None:\n"
            + "            raise IntegrationConfigurationError(\n"
            + f'                "{label} integration requires a runtime delegate for catalog operations",\n'
            + "            )\n"
            + "        return runtime\n"
        )

    return (
        "    _runtime: Any | None = PrivateAttr(default=None)\n\n"
        "    @classmethod\n"
        f"    def from_runtime(cls, runtime: Any, *, enabled: bool = True) -> {prefix}Integration:\n"
        "        integration = cls.for_provider(\n"
        f"            provider_id={const},\n"
        f'            display_name="{label}",\n'
        f"            config={prefix}IntegrationConfig(enabled=enabled),\n"
        "        )\n"
        "        integration._runtime = runtime\n"
        "        return integration\n\n"
        "    def _require_runtime(self) -> Any:\n"
        "        if self._runtime is None:\n"
        f'            raise IntegrationConfigurationError("{label} integration requires a runtime delegate")\n'
        "        return self._runtime\n\n"
        "    def __getattr__(self, name: str) -> Any:\n"
        "        if name.startswith('_'):\n"
        "            raise AttributeError(name)\n"
        "        return getattr(self._require_runtime(), name)\n"
    )


def generate_integration_py(slug: str, category: str, legacy_factory: str) -> str:
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    label = display_name(slug)
    cat_label = category.replace("_", " ")

    if category == "observability_backend":
        existing = (pkg_path(slug, category) / "integration.py").read_text(encoding="utf-8")
        signals_const, transport_name = _parse_observability_constants(existing)
        transport_name = transport_name or f"{prefix}Transport"
        signals_const = signals_const or f"{slug.upper()}_SUPPORTED_SIGNALS"
        preamble_match = re.search(
            rf"({const} = \"{slug}\"[\s\S]*?{signals_const} = _\w+)",
            existing,
        )
        signals_preamble = preamble_match.group(1) if preamble_match else (
            f'{const} = "{slug}"\n\n{signals_const} = ()'
        )
        spec = CATEGORY_RUNTIME_SPECS["observability_backend"]
        return (
            H
            + f'"""{label} observability vendor integration (INTEGRATIONS-2C · INTEGRATIONS-2E runtime cutover)."""\n\n'
            + "from __future__ import annotations\n\n"
            + "from typing import Any, Protocol, runtime_checkable\n\n"
            + "from pydantic import PrivateAttr\n\n"
            + "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
            + f"{spec['protocol_import']}\n"
            + "from intergrax.runtime.integrations.observability import (\n"
            + "    ObservabilityVendorIntegrationConfig,\n"
            + "    ObservabilityVendorIntegrationContract,\n"
            + "    ObservabilityVendorPayload,\n"
            + "    ObservabilityVendorSignal,\n"
            + ")\n\n"
            + signals_preamble
            + "\n\n"
            + f"class {prefix}IntegrationConfig(ObservabilityVendorIntegrationConfig):\n"
            + f'    """Typed config for {label} observability vendor integration."""\n\n'
            + "    pass\n\n\n"
            + "@runtime_checkable\n"
            + f"class {transport_name}(Protocol):\n"
            + '    """Injectable delivery facade — no vendor SDK or network I/O in the integration class."""\n\n'
            + "    async def send_observability_payload(self, payload: ObservabilityVendorPayload) -> None:\n"
            + f'        """Deliver a policy-sanitized vendor payload to {label}."""\n\n\n'
            + f"class {prefix}Integration(ObservabilityVendorIntegrationContract):\n"
            + '    """\n'
            + f"    Single public {label} observability entrypoint.\n\n"
            + f"    Legacy catalog factory ({legacy_factory}) delegates to this class via from_backend().\n"
            + '    """\n\n'
            + f"    config: {prefix}IntegrationConfig = {prefix}IntegrationConfig()\n"
            + f"    _transport: {transport_name} | None = PrivateAttr(default=None)\n"
            + _runtime_block(category, prefix, slug, const, label, signals_const=signals_const)
            + "\n\n"
            + "    @classmethod\n"
            + "    def from_transport(\n"
            + "        cls,\n"
            + f"        transport: {transport_name},\n"
            + "        *,\n"
            + "        enabled: bool = False,\n"
            + "        supported_signals: tuple[ObservabilityVendorSignal, ...] | None = None,\n"
            + f"    ) -> {prefix}Integration:\n"
            + "        integration = cls.for_provider(\n"
            + f"            provider_id={const},\n"
            + f"            supported_signals=supported_signals or {signals_const},\n"
            + f'            display_name="{label}",\n'
            + f"            config={prefix}IntegrationConfig(enabled=enabled),\n"
            + "        )\n"
            + "        integration._transport = transport\n"
            + "        return integration\n\n"
            + "    @property\n"
            + f"    def transport(self) -> {transport_name} | None:\n"
            + "        return self._transport\n\n"
            + "    async def deliver_payload(self, payload: ObservabilityVendorPayload) -> None:\n"
            + "        if self._transport is None:\n"
            + f'            msg = "{prefix}Integration requires an injected transport for delivery"\n'
            + "            raise RuntimeError(msg)\n"
            + "        await self._transport.send_observability_payload(payload)\n\n"
            + f"{spec['protocol_name']}.register({prefix}Integration)\n"
        )

    contract_module, contract_class = _CATEGORY_CONTRACT[category]
    spec = CATEGORY_RUNTIME_SPECS.get(category, {})
    typing_extra = spec.get("typing_imports", "")
    typing_line = f"from typing import Any, Protocol, Sequence, {typing_extra}runtime_checkable\n"
    extra_imports = spec.get("extra_imports", "")
    protocol_import = spec.get("protocol_import", "")
    register_line = ""
    if protocol_import and spec.get("register_protocol"):
        register_line = f"\n{spec['protocol_name']}.register({prefix}Integration)\n"

    body = (
        H
        + f'"""{label} {cat_label} integration (INTEGRATIONS-2D · INTEGRATIONS-2E runtime cutover)."""\n\n'
        + "from __future__ import annotations\n\n"
        + typing_line
        + "\nfrom pydantic import PrivateAttr\n\n"
        + "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
        + (f"{extra_imports}\n" if extra_imports else "")
        + (f"{protocol_import}\n" if protocol_import else "")
        + f"from {contract_module} import {contract_class}\n"
        + "from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig\n\n"
        + f'{const} = "{slug}"\n\n\n'
        + f"class {prefix}IntegrationConfig(CategoryIntegrationConfig):\n"
        + f'    """Typed config for {label} {cat_label} integration."""\n\n'
        + "    pass\n\n\n"
        + "@runtime_checkable\n"
        + f"class {prefix}Client(Protocol):\n"
        + '    """Injectable client facade — no vendor SDK or network I/O in the integration class."""\n\n'
        + "    async def ping(self) -> None:\n"
        + '        """Lightweight connectivity check."""\n\n\n'
        + f"class {prefix}Integration({contract_class}):\n"
        + '    """\n'
        + f"    Single public {label} {cat_label} entrypoint.\n\n"
        + f"    Legacy catalog factory ({legacy_factory}) delegates to this class.\n"
        + '    """\n\n'
        + f"    config: {prefix}IntegrationConfig = {prefix}IntegrationConfig()\n"
        + f"    _client: {prefix}Client | None = PrivateAttr(default=None)\n"
        + _runtime_block(category, prefix, slug, const, label)
        + "\n\n"
        + "    @classmethod\n"
        + "    def from_client(\n"
        + "        cls,\n"
        + f"        client: {prefix}Client,\n"
        + "        *,\n"
        + "        enabled: bool = False,\n"
        + f"    ) -> {prefix}Integration:\n"
        + "        integration = cls.for_provider(\n"
        + f"            provider_id={const},\n"
        + f'            display_name="{label}",\n'
        + f"            config={prefix}IntegrationConfig(enabled=enabled),\n"
        + "        )\n"
        + "        integration._client = client\n"
        + "        return integration\n\n"
        + "    @property\n"
        + f"    def client(self) -> {prefix}Client | None:\n"
        + "        return self._client\n"
        + register_line
    )
    return body


def _wrap_legacy_factory(source: str, legacy_name: str, prefix: str) -> str:
    if f"def {legacy_name}(" in source and "_legacy_" in source:
        return source
    import_patterns = [
        r"from intergrax\.integrations\._shared\.p[0-9]\.factories import ([^\n]+)",
        r"from intergrax\.integrations\._shared\.p[0-9]\.factories import (\([^)]+\))",
    ]
    legacy_import = None
    for pattern in import_patterns:
        match = re.search(pattern, source)
        if match:
            legacy_import = match.group(0)
            break
    if legacy_import and f" as _legacy_{legacy_name}" not in source:
        names = legacy_import.split("import", 1)[1].strip()
        if names == legacy_name:
            source = source.replace(
                legacy_import,
                legacy_import.replace(legacy_name, f"{legacy_name} as _legacy_{legacy_name}"),
            )

    if f"def {legacy_name}(" not in source:
        integration_cls = f"{prefix}Integration"
        wrapper = (
            f"\n\nfrom {source.split('from intergrax.integrations.providers.')[1].split('.integration')[0]}"
            if False
            else ""
        )
        # find integration import module
        integ_import = re.search(
            rf"from intergrax\.integrations\.providers\.[\w.]+\.integration import \([\s\S]*?{integration_cls}",
            source,
        )
        if not integ_import and f"{integration_cls}" not in source:
            pkg_match = re.search(r"intergrax/integrations/providers/([\w/]+)/bundle.py", str(source))
        wrapper = (
            f"\n\ndef {legacy_name}(**kwargs: object) -> {integration_cls}:\n"
            f'    """Compatibility shim — wraps legacy runtime in {integration_cls}."""\n'
            f"    runtime = _legacy_{legacy_name}(**kwargs)\n"
            f"    if isinstance(runtime, {integration_cls}):\n"
            f"        return runtime\n"
            f"    return {integration_cls}.from_runtime(runtime)\n"
        )
        if f"_legacy_{legacy_name}" in source and f"def {legacy_name}(" not in source:
            source = source.rstrip() + wrapper
    return source


def patch_reexport_bundle(slug: str, category: str, legacy_name: str, prefix: str) -> None:
    path = pkg_path(slug, category) / "bundle.py"
    source = path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    if f"Compatibility shim" in source and f"def {legacy_name}(" in source:
        return

    # Rename direct import to _legacy_
    for line in source.splitlines():
        if "from intergrax.integrations._shared.p" in line and "factories import" in line:
            if f" as _legacy_{legacy_name}" in line:
                break
            if legacy_name in line and " as " not in line:
                source = source.replace(
                    f"import {legacy_name}",
                    f"import {legacy_name} as _legacy_{legacy_name}",
                )
                break

    if f"def {legacy_name}(" not in source:
        if f"from intergrax.integrations.providers.{category}.{slug}.integration import" not in source:
            const = provider_id_const(slug, category)
            source += (
                f"\nfrom intergrax.integrations.providers.{category}.{slug}.integration import "
                f"{integration_cls}\n"
            )
        from_runtime = "from_store" if category == "vector_store" else "from_runtime"
        if category == "observability_backend":
            from_runtime = "from_backend"
        source += (
            f"\n\ndef {legacy_name}(**kwargs: object) -> {integration_cls}:\n"
            f'    """Compatibility shim — constructs {integration_cls} from legacy runtime."""\n'
            f"    runtime = _legacy_{legacy_name}(**kwargs)\n"
            f"    if isinstance(runtime, {integration_cls}):\n"
            f"        return runtime\n"
            f"    return {integration_cls}.{from_runtime}(runtime)\n"
        )
    path.write_text(source, encoding="utf-8")


def patch_vector_store_opens(slug: str, category: str, prefix: str) -> None:
    opens_path = pkg_path(slug, category) / "opens.py"
    if not opens_path.is_file():
        return
    source = opens_path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    source = re.sub(
        rf"from intergrax\.integrations\.providers\.{category}\.{slug}\.adapter import {integration_cls}",
        f"from intergrax.integrations.providers.{category}.{slug}.integration import {integration_cls}",
        source,
    )
    source = source.replace(
        f"return {integration_cls}(config, inner)",
        f"return {integration_cls}.from_store(config, inner)",
    )
    opens_path.write_text(source, encoding="utf-8")


def patch_vector_store_bundle(slug: str, category: str, prefix: str) -> None:
    bundle_path = pkg_path(slug, category) / "bundle.py"
    if not bundle_path.is_file():
        return
    source = bundle_path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    source = re.sub(
        rf"from intergrax\.integrations\.providers\.{category}\.{slug}\.adapter import {integration_cls}",
        f"from intergrax.integrations.providers.{category}.{slug}.integration import {integration_cls}",
        source,
    )
    # Move contract imports to top if duplicated at bottom - minimal fix
    if f"The legacy facade" in source:
        source = source.replace(
            "The legacy facade (create_",
            "Compatibility shim — constructs Integration via from_store (create_",
        )
    bundle_path.write_text(source, encoding="utf-8")


def privatize_adapter(slug: str, category: str, prefix: str) -> None:
    adapter_path = pkg_path(slug, category) / "adapter.py"
    if not adapter_path.is_file():
        return
    source = adapter_path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == integration_cls:
            adapter_path.unlink()
            return
    # Rename public adapter classes to private
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            source = source.replace(f"class {node.name}", f"class _{node.name}")
    adapter_path.write_text(source, encoding="utf-8")


def wrap_bundle_legacy_factory(slug: str, category: str, legacy_name: str, prefix: str) -> None:
    path = pkg_path(slug, category) / "bundle.py"
    if not path.is_file():
        return
    source = path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    from_method = (
        "from_store" if category == "vector_store" else "from_backend" if category == "observability_backend" else "from_runtime"
    )
    if f"{integration_cls}.{from_method}(" in source and f"def {legacy_name}(" in source:
        return
    lines = source.splitlines()
    out: list[str] = []
    in_fn = False
    fn_indent = ""
    wrapped = False
    for line in lines:
        if line.startswith(f"def {legacy_name}("):
            in_fn = True
            fn_indent = "    "
            if f" -> {integration_cls}" not in line and " -> " in line:
                line = re.sub(r" -> [^:]+:", f" -> {integration_cls}:", line)
            elif " -> " not in line and line.rstrip().endswith(":"):
                line = line[:-1] + f" -> {integration_cls}:"
        elif in_fn and (line.startswith("def ") or (line and not line.startswith(" ") and not line.startswith("\t"))):
            in_fn = False
        if in_fn and line.strip().startswith("return ") and integration_cls not in line:
            expr = line.strip()[len("return ") :]
            line = f"{fn_indent}return {integration_cls}.{from_method}({expr})"
            wrapped = True
        out.append(line)
    if wrapped:
        if f"from intergrax.integrations.providers.{category}.{slug}.integration import {integration_cls}" not in source:
            out.insert(
                0,
                f"from intergrax.integrations.providers.{category}.{slug}.integration import {integration_cls}  # noqa: E402",
            )
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


def patch_named_adapter_opens(slug: str, category: str, prefix: str) -> None:
    adapter_path = pkg_path(slug, category) / "adapter.py"
    opens_path = pkg_path(slug, category) / "opens.py"
    if not adapter_path.is_file() or not opens_path.is_file():
        return
    source = opens_path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    adapter_source = adapter_path.read_text(encoding="utf-8")
    private_cls = None
    for line in adapter_source.splitlines():
        if line.startswith("class _"):
            private_cls = line.split("(")[0].replace("class ", "").strip()
            break
    if private_cls is None:
        return
    source = re.sub(
        rf"from intergrax\.integrations\.providers\.{category}\.{slug}\.adapter import \w+",
        f"from intergrax.integrations.providers.{category}.{slug}.adapter import {private_cls}",
        source,
    )
    if f"return {integration_cls}." not in source:
        source = re.sub(
            rf"return {private_cls}\(([^)]*)\)",
            rf"return {integration_cls}.from_runtime({private_cls}(\1))",
            source,
        )
    opens_path.write_text(source, encoding="utf-8")


def patch_opens_adapter_imports(slug: str, category: str, prefix: str) -> None:
    opens_path = pkg_path(slug, category) / "opens.py"
    if not opens_path.is_file():
        return
    source = opens_path.read_text(encoding="utf-8")
    integration_cls = f"{prefix}Integration"
    adapter_path = pkg_path(slug, category) / "adapter.py"
    if adapter_path.is_file():
        adapter_source = adapter_path.read_text(encoding="utf-8")
        for line in adapter_source.splitlines():
            if line.startswith("class _"):
                private_cls = line.split("(")[0].replace("class ", "").strip()
                if f"from intergrax.integrations.providers.{category}.{slug}.adapter import" in source:
                    source = re.sub(
                        rf"from intergrax\.integrations\.providers\.{category}\.{slug}\.adapter import \w+",
                        f"from intergrax.integrations.providers.{category}.{slug}.adapter import {private_cls}",
                        source,
                    )
                if f"return {integration_cls}(" in source:
                    source = source.replace(
                        f"return {integration_cls}(",
                        f"return {prefix}Integration.from_runtime({private_cls}(",
                    ).replace("))", "))", 1)
    opens_path.write_text(source, encoding="utf-8")


def update_usage_md(slug: str, category: str, prefix: str) -> None:
    path = pkg_path(slug, category) / "USAGE.md"
    if not path.is_file():
        return
    label = display_name(slug)
    text = path.read_text(encoding="utf-8")
    if "Single public entrypoint" in text:
        return
    path.write_text(
        f"# {label} ({slug})\n\n"
        f"Category: `{category}`\n\n"
        f"## Single public entrypoint\n\n"
        f"- **`{prefix}Integration`** in `integration.py` is the only public provider class.\n"
        f"- Legacy catalog factories are compatibility shims delegating to `{prefix}Integration`.\n"
        f"- Contract factory: `create_{slug}_{category}_integration()`.\n",
        encoding="utf-8",
    )


def process_slug(slug: str, category: str, *, dry_run: bool) -> bool:
    if slug in DEFERRED_SLUGS or slug in ALREADY_CUTOVER:
        return False
    pkg = pkg_path(slug, category)
    integration_path = pkg / "integration.py"
    if not integration_path.is_file():
        return False
    if is_cutover_done(integration_path):
        return False
    prefix = class_prefix(slug, category)
    legacy_factory = detect_legacy_factory(slug, category, pkg)
    new_integration = generate_integration_py(slug, category, legacy_factory)
    if dry_run:
        print(f"would cut over {category}/{slug}")
        return True
    integration_path.write_text(new_integration, encoding="utf-8")
    if (pkg / "bundle.py").is_file():
        bundle_src = (pkg / "bundle.py").read_text(encoding="utf-8")
        if "from intergrax.integrations._shared.p" in bundle_src and "factories import" in bundle_src:
            patch_reexport_bundle(slug, category, legacy_factory, prefix)
        else:
            wrap_bundle_legacy_factory(slug, category, legacy_factory, prefix)
        if category == "vector_store":
            patch_vector_store_bundle(slug, category, prefix)
            patch_vector_store_opens(slug, category, prefix)
        elif (pkg / "adapter.py").is_file():
            patch_vector_store_bundle(slug, category, prefix)
    privatize_adapter(slug, category, prefix)
    patch_named_adapter_opens(slug, category, prefix)
    patch_opens_adapter_imports(slug, category, prefix)
    update_usage_md(slug, category, prefix)
    print(f"cut over {category}/{slug}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--slug", action="append", default=[])
    args = parser.parse_args()
    slugs = args.slug or [
        s for s, c in SLUG_CATEGORY.items() if s not in DEFERRED_SLUGS and s not in ALREADY_CUTOVER
    ]
    count = 0
    for slug in sorted(slugs):
        category = SLUG_CATEGORY[slug]
        if category == "llm_guardrail":
            continue
        if process_slug(slug, category, dry_run=args.dry_run):
            count += 1
    print(f"processed {count} slugs")


if __name__ == "__main__":
    main()
