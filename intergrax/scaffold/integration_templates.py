# © Artur Czarnecki. All rights reserved.

"""Code templates for ``intergrax.scaffold new-integration`` (INTEGRATIONS-2E layout)."""

from __future__ import annotations

from intergrax.integrations._shared.runtime_cutover_templates import (
    CATEGORY_RUNTIME_SPECS,
    GENERIC_RUNTIME_CATEGORIES,
)
from intergrax.runtime.integrations.categories import PROVIDER_CATEGORY_CONTRACT_REGISTRY

_HEADER = (
    "# © Artur Czarnecki. All rights reserved.\n"
    "# Intergrax framework – proprietary and confidential.\n\n"
)

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


def catalog_factory_name(slug: str, category: str) -> str:
    return f"create_{slug}_{category}"


def contract_factory_name(slug: str, category: str) -> str:
    if category == "observability_backend":
        return f"create_{slug}_observability_integration"
    return f"create_{slug}_{category}_integration"


def validate_category(category: str) -> str | None:
    if category not in PROVIDER_CATEGORY_CONTRACT_REGISTRY:
        known = ", ".join(sorted(PROVIDER_CATEGORY_CONTRACT_REGISTRY))
        return f"error: unknown category {category!r} (expected one of: {known})"
    return None


def _require_method(runtime_attr: str) -> tuple[str, str]:
    if runtime_attr == "_inner":
        return "_require_inner", "inner store"
    return "_require_client", "injected client"


def _delegation_block(category: str, prefix: str, label: str) -> str:
    spec = CATEGORY_RUNTIME_SPECS.get(category, {})
    runtime_attr = spec.get("runtime_attr", "_client")
    require_name, require_target = _require_method(str(runtime_attr))

    if category in GENERIC_RUNTIME_CATEGORIES or not spec.get("methods"):
        return (
            f"    def {require_name}(self) -> Any:\n"
            f"        client = self.{runtime_attr}\n"
            "        if client is None:\n"
            "            raise IntegrationConfigurationError(\n"
            f'                f"{{type(self).__name__}} requires an {require_target} for operations",\n'
            "            )\n"
            "        return client\n\n"
            "    def __getattr__(self, name: str) -> Any:\n"
            '        if name.startswith("_"):\n'
            "            raise AttributeError(name)\n"
            f"        return getattr(self.{require_name}(), name)\n"
        )

    extra_properties = spec.get("extra_properties", "")
    methods = spec.get("methods", "")
    return (
        f"{extra_properties}"
        f"{methods}\n\n"
        f"    def {require_name}(self) -> Any:\n"
        f"        client = self.{runtime_attr}\n"
        "        if client is None:\n"
        "            raise IntegrationConfigurationError(\n"
        f'                f"{{type(self).__name__}} requires an {require_target} for operations",\n'
        "            )\n"
        "        return client\n"
    )


def generate_integration_py(slug: str, category: str) -> str:
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    label = display_name(slug)
    cat_label = category.replace("_", " ")
    legacy_factory = catalog_factory_name(slug, category)

    if category == "observability_backend":
        transport_name = f"{prefix}Transport"
        signals_const = f"{slug.upper()}_SUPPORTED_SIGNALS"
        return (
            _HEADER
            + f'"""{label} observability vendor integration scaffold (INTEGRATIONS-2E)."""\n\n'
            + "from __future__ import annotations\n\n"
            + "from typing import Any, Protocol, runtime_checkable\n\n"
            + "from pydantic import PrivateAttr\n\n"
            + "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
            + "from intergrax.runtime.integrations.observability import (\n"
            + "    ObservabilityVendorIntegrationConfig,\n"
            + "    ObservabilityVendorIntegrationContract,\n"
            + "    ObservabilityVendorPayload,\n"
            + "    ObservabilityVendorSignal,\n"
            + ")\n\n"
            + f'{const} = "{slug}"\n\n'
            + f"{signals_const}: tuple[ObservabilityVendorSignal, ...] = ()\n\n\n"
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
            + f"    Wire catalog behavior in ``{legacy_factory}``; inject transport via ``from_transport()``.\n"
            + '    """\n\n'
            + f"    config: {prefix}IntegrationConfig = {prefix}IntegrationConfig()\n"
            + f"    _transport: {transport_name} | None = PrivateAttr(default=None)\n\n"
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
            + "            raise IntegrationConfigurationError(\n"
            + f'                "{prefix}Integration requires an injected transport for delivery",\n'
            + "            )\n"
            + "        await self._transport.send_observability_payload(payload)\n"
        )

    contract_cls = PROVIDER_CATEGORY_CONTRACT_REGISTRY[category]
    contract_module = contract_cls.__module__
    contract_class = contract_cls.__name__
    spec = CATEGORY_RUNTIME_SPECS.get(category, {})
    runtime_attr = spec.get("runtime_attr", "_client")
    typing_extra = spec.get("typing_imports", "")
    typing_line = f"from typing import Any, Protocol, {typing_extra}runtime_checkable\n"
    extra_imports = spec.get("extra_imports", "")
    protocol_import = spec.get("protocol_import", "")
    protocol_name = spec.get("protocol_name", "")
    register_protocol = bool(spec.get("register_protocol"))

    client_block = ""
    if register_protocol and protocol_name:
        client_block = f"{prefix}Client = {protocol_name}\n\n\n"
    else:
        client_block = (
            "@runtime_checkable\n"
            + f"class {prefix}Client(Protocol):\n"
            + '    """Injectable client facade — no vendor SDK or network I/O in the integration class."""\n\n'
            + "    async def ping(self) -> None:\n"
            + '        """Lightweight connectivity check."""\n\n\n'
        )

    register_line = ""
    if register_protocol and protocol_name:
        register_line = f"\n{protocol_name}.register({prefix}Integration)\n"

    return (
        _HEADER
        + f'"""{label} {cat_label} integration scaffold (INTEGRATIONS-2E)."""\n\n'
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
        + client_block
        + f"class {prefix}Integration({contract_class}):\n"
        + '    """\n'
        + f"    Single public {label} {cat_label} entrypoint.\n\n"
        + f"    Wire catalog behavior in ``{legacy_factory}``; inject clients via ``from_client()``.\n"
        + '    """\n\n'
        + f"    config: {prefix}IntegrationConfig = {prefix}IntegrationConfig()\n"
        + f"    {runtime_attr}: {prefix}Client | None = PrivateAttr(default=None)\n\n"
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
        + f"        integration.{runtime_attr} = client\n"
        + "        return integration\n\n"
        + "    @property\n"
        + f"    def client(self) -> {prefix}Client | None:\n"
        + f"        return self.{runtime_attr}\n\n"
        + _delegation_block(category, prefix, label)
        + register_line
    )


def generate_bundle_py(slug: str, category: str) -> str:
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    label = display_name(slug)
    cat_label = category.replace("_", " ")
    catalog_factory = catalog_factory_name(slug, category)
    contract_factory = contract_factory_name(slug, category)

    if category == "observability_backend":
        signals_const = f"{slug.upper()}_SUPPORTED_SIGNALS"
        transport_name = f"{prefix}Transport"
        return (
            _HEADER
            + "from __future__ import annotations\n\n"
            + "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
            + f"from intergrax.integrations.providers.{category}.{slug}.integration import (\n"
            + f"    {const},\n"
            + f"    {signals_const},\n"
            + f"    {prefix}Integration,\n"
            + f"    {prefix}IntegrationConfig,\n"
            + f"    {transport_name},\n"
            + ")\n\n"
            + "__all__ = [\n"
            + f'    "{catalog_factory}",\n'
            + f'    "{contract_factory}",\n'
            + "]\n\n\n"
            + f"def {contract_factory}(\n"
            + "    *,\n"
            + f"    transport: {transport_name} | None = None,\n"
            + "    enabled: bool = False,\n"
            + f") -> {prefix}Integration:\n"
            + '    """Build a contract-based observability vendor integration."""\n'
            + "    if enabled and transport is None:\n"
            + "        raise IntegrationConfigurationError(\n"
            + f'            "{label} observability integration requires an injected transport when enabled=True",\n'
            + "        )\n"
            + "    if transport is not None:\n"
            + "        return "
            + f"{prefix}Integration.from_transport(transport, enabled=enabled)\n"
            + f"    return {prefix}Integration.for_provider(\n"
            + f"        provider_id={const},\n"
            + f"        supported_signals={signals_const},\n"
            + f'        display_name="{label}",\n'
            + f"        config={prefix}IntegrationConfig(enabled=enabled),\n"
            + "    )\n\n\n"
            + f"def {catalog_factory}(**kwargs: object) -> {prefix}Integration:\n"
            + '    """Catalog factory — wire vendor SDK and return a configured integration."""\n'
            + "    _ = kwargs\n"
            + f"    return {contract_factory}(enabled=False)\n"
        )

    return (
        _HEADER
        + "from __future__ import annotations\n\n"
        + "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
        + f"from intergrax.integrations.providers.{category}.{slug}.integration import (\n"
        + f"    {const},\n"
        + f"    {prefix}Integration,\n"
        + f"    {prefix}IntegrationConfig,\n"
        + f"    {prefix}Client,\n"
        + ")\n\n"
        + "__all__ = [\n"
        + f'    "{catalog_factory}",\n'
        + f'    "{contract_factory}",\n'
        + "]\n\n\n"
        + f"def {contract_factory}(\n"
        + "    *,\n"
        + f"    client: {prefix}Client | None = None,\n"
        + "    enabled: bool = False,\n"
        + f") -> {prefix}Integration:\n"
        + f'    """Build a contract-based {label} {cat_label} integration."""\n'
        + "    if enabled and client is None:\n"
        + "        raise IntegrationConfigurationError(\n"
        + f'            "{label} {cat_label} integration requires an injected client when enabled=True",\n'
        + "        )\n"
        + "    if client is not None:\n"
        + f"        return {prefix}Integration.from_client(client, enabled=enabled)\n"
        + f"    return {prefix}Integration.for_provider(\n"
        + f"        provider_id={const},\n"
        + f'        display_name="{label}",\n'
        + f"        config={prefix}IntegrationConfig(enabled=enabled),\n"
        + "    )\n\n\n"
        + f"def {catalog_factory}(**kwargs: object) -> {prefix}Integration:\n"
        + '    """Catalog factory — wire vendor SDK and return a configured integration."""\n'
        + "    _ = kwargs\n"
        + f"    return {contract_factory}(enabled=False)\n"
    )


def generate_register_py(slug: str, category: str) -> str:
    catalog_factory = catalog_factory_name(slug, category)
    return (
        _HEADER
        + f'"""Register {slug} in the integration catalog."""\n\n'
        + "from __future__ import annotations\n\n"
        + f"from intergrax.integrations.providers.{category}.{slug}.bundle import {catalog_factory}\n"
        + f"from intergrax.integrations.providers.{category}.{slug}.manifest import MANIFEST\n"
        + "from intergrax.integrations.registry.plugin_register import register_from_manifest\n\n\n"
        + f"def register_{slug}_integration(*, override: bool = False) -> None:\n"
        + f"    register_from_manifest(MANIFEST, {catalog_factory}, override=override)\n"
    )


def generate_usage_md(slug: str, category: str) -> str:
    prefix = class_prefix(slug, category)
    catalog_factory = catalog_factory_name(slug, category)
    contract_factory = contract_factory_name(slug, category)
    return (
        f"# `{slug}` integration — usage\n\n"
        f"**Category:** ``{category}``\n\n"
        + "## Contract entrypoint\n\n"
        + "```python\n"
        + f"from intergrax.integrations.providers.{category}.{slug}.integration import {prefix}Integration\n"
        + f"from intergrax.integrations.providers.{category}.{slug}.bundle import {contract_factory}\n\n"
        + f"integration = {contract_factory}()\n"
        + "```\n\n"
        + "## Catalog profile resolution\n\n"
        + "```python\n"
        + "from intergrax.integrations.contracts.base import IntegrationCategory\n"
        + "from intergrax.integrations.registry.bootstrap import register_default_integrations\n"
        + "from intergrax.integrations.registry.profile import IntegrationProfile\n\n"
        + "register_default_integrations()\n"
        + f'profile = IntegrationProfile({category}="{slug}")\n'
        + f"backend = profile.resolve(IntegrationCategory.{category.upper()})\n"
        + "```\n\n"
        + "## Catalog registration\n\n"
        + "```python\n"
        + f"from intergrax.integrations.providers.{category}.{slug}.register import register_{slug}_integration\n\n"
        + f"register_{slug}_integration()\n"
        + "```\n\n"
        + f"Implement vendor wiring in ``{catalog_factory}`` and optional ``opens.py`` when needed.\n"
    )
