#!/usr/bin/env python3
"""Generate contract-based provider integration packages (INTEGRATIONS-2D)."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from intergrax.integrations.providers.layout import SLUG_CATEGORY  # noqa: E402

H = "# © Artur Czarnecki. All rights reserved.\n# Intergrax framework – proprietary and confidential.\n\n"

WAVE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "w1": ("vector_store", "search_provider", "document_parser", "rerank_provider", "wiki_knowledge"),
    "w2": ("relational_store", "document_store", "key_value_cache", "graph_store", "object_storage"),
    "w3": (
        "message_bus",
        "notification_channel",
        "interaction_surface",
        "collaboration_suite",
        "issue_tracker",
        "browser_automation",
    ),
    "w4": (
        "cloud_platform",
        "secrets_store",
        "feature_flag",
        "ci_cd",
        "security_scanner",
        "sandbox_host",
        "identity_provider",
        "workflow_orchestrator",
    ),
    "w5": (
        "speech_provider",
        "vision_serving",
        "ml_inference_host",
        "llm_guardrail",
        "billing_meter",
        "crm",
    ),
}

REGISTER_FACTORY_SLUGS: frozenset[str] = frozenset({"yt_dlp"})

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

_DISPLAY_OVERRIDES: dict[str, str] = {
    "aws": "AWS",
    "gcp": "GCP",
    "azure": "Azure",
    "pgvector": "pgvector",
    "yt_dlp": "yt-dlp",
    "n8n": "n8n",
    "e2b": "E2B",
    "okta": "Okta",
    "auth0": "Auth0",
    "ci_cd": "CI/CD",
}

CATEGORY_CONTRACT: dict[str, tuple[str, str]] = {
    "relational_store": (
        "intergrax.runtime.integrations.categories.data",
        "RelationalStoreIntegrationContract",
    ),
    "document_store": (
        "intergrax.runtime.integrations.categories.data",
        "DocumentStoreIntegrationContract",
    ),
    "key_value_cache": (
        "intergrax.runtime.integrations.categories.data",
        "KeyValueCacheIntegrationContract",
    ),
    "graph_store": (
        "intergrax.runtime.integrations.categories.data",
        "GraphStoreIntegrationContract",
    ),
    "message_bus": (
        "intergrax.runtime.integrations.categories.messaging",
        "MessageBusIntegrationContract",
    ),
    "notification_channel": (
        "intergrax.runtime.integrations.categories.messaging",
        "NotificationChannelIntegrationContract",
    ),
    "object_storage": (
        "intergrax.runtime.integrations.categories.storage",
        "ObjectStorageIntegrationContract",
    ),
    "vector_store": (
        "intergrax.runtime.integrations.categories.storage",
        "VectorStoreIntegrationContract",
    ),
    "search_provider": (
        "intergrax.runtime.integrations.categories.search",
        "SearchProviderIntegrationContract",
    ),
    "rerank_provider": (
        "intergrax.runtime.integrations.categories.search",
        "RerankProviderIntegrationContract",
    ),
    "interaction_surface": (
        "intergrax.runtime.integrations.categories.collaboration",
        "InteractionSurfaceIntegrationContract",
    ),
    "collaboration_suite": (
        "intergrax.runtime.integrations.categories.collaboration",
        "CollaborationSuiteIntegrationContract",
    ),
    "issue_tracker": (
        "intergrax.runtime.integrations.categories.collaboration",
        "IssueTrackerIntegrationContract",
    ),
    "wiki_knowledge": (
        "intergrax.runtime.integrations.categories.collaboration",
        "WikiKnowledgeIntegrationContract",
    ),
    "browser_automation": (
        "intergrax.runtime.integrations.categories.automation",
        "BrowserAutomationIntegrationContract",
    ),
    "billing_meter": (
        "intergrax.runtime.integrations.categories.automation",
        "BillingMeterIntegrationContract",
    ),
    "crm": (
        "intergrax.runtime.integrations.categories.automation",
        "CrmIntegrationContract",
    ),
    "cloud_platform": (
        "intergrax.runtime.integrations.categories.devops",
        "CloudPlatformIntegrationContract",
    ),
    "ci_cd": (
        "intergrax.runtime.integrations.categories.devops",
        "CiCdIntegrationContract",
    ),
    "security_scanner": (
        "intergrax.runtime.integrations.categories.devops",
        "SecurityScannerIntegrationContract",
    ),
    "sandbox_host": (
        "intergrax.runtime.integrations.categories.devops",
        "SandboxHostIntegrationContract",
    ),
    "workflow_orchestrator": (
        "intergrax.runtime.integrations.categories.devops",
        "WorkflowOrchestratorIntegrationContract",
    ),
    "secrets_store": (
        "intergrax.runtime.integrations.categories.security",
        "SecretsStoreIntegrationContract",
    ),
    "feature_flag": (
        "intergrax.runtime.integrations.categories.security",
        "FeatureFlagIntegrationContract",
    ),
    "identity_provider": (
        "intergrax.runtime.integrations.categories.security",
        "IdentityProviderIntegrationContract",
    ),
    "speech_provider": (
        "intergrax.runtime.integrations.categories.ai",
        "SpeechProviderIntegrationContract",
    ),
    "vision_serving": (
        "intergrax.runtime.integrations.categories.ai",
        "VisionServingIntegrationContract",
    ),
    "ml_inference_host": (
        "intergrax.runtime.integrations.categories.ai",
        "MlInferenceHostIntegrationContract",
    ),
    "document_parser": (
        "intergrax.runtime.integrations.categories.ai",
        "DocumentParserIntegrationContract",
    ),
    "llm_guardrail": (
        "intergrax.runtime.integrations.categories.ai",
        "LlmGuardrailIntegrationContract",
    ),
}


def slug_to_pascal(slug: str) -> str:
    if slug in _CLASS_NAME_OVERRIDES:
        return _CLASS_NAME_OVERRIDES[slug]
    return "".join(part.capitalize() for part in slug.split("_"))


def category_to_pascal(category: str) -> str:
    return "".join(part.capitalize() for part in category.split("_"))


def display_name(slug: str) -> str:
    if slug in _DISPLAY_OVERRIDES:
        return _DISPLAY_OVERRIDES[slug]
    return slug.replace("_", " ").title()


def provider_id_const(slug: str, category: str) -> str:
    return f"{slug.upper()}_{category.upper()}_PROVIDER_ID"


def class_prefix(slug: str, category: str) -> str:
    return f"{slug_to_pascal(slug)}{category_to_pascal(category)}"


def contract_factory_name(slug: str, category: str) -> str:
    return f"create_{slug}_{category}_integration"


def _contract_factory_exists(source: str, slug: str, category: str) -> bool:
    return contract_factory_name(slug, category) in source


def detect_legacy_factory(slug: str, category: str, pkg: Path) -> str:
    bundle_path = pkg / "bundle.py"
    register_path = pkg / "register.py"
    candidates: list[str] = []
    if bundle_path.is_file():
        src = bundle_path.read_text(encoding="utf-8")
        match = re.search(r"__all__\s*=\s*\[(.*?)\]", src, re.S)
        if match:
            candidates.extend(re.findall(r'"(create_[^"]+)"', match.group(1)))
            candidates.extend(re.findall(r"'(create_[^']+)'", match.group(1)))
        if not candidates:
            candidates.extend(re.findall(r"def (create_\w+)\(", src))
    if register_path.is_file() and not candidates:
        src = register_path.read_text(encoding="utf-8")
        candidates.extend(re.findall(r"def (create_\w+)\(", src))

    contract_name = contract_factory_name(slug, category)
    legacy = [name for name in candidates if name != contract_name]
    if not legacy:
        msg = f"{slug}: no legacy factory found"
        raise RuntimeError(msg)

    preferred_prefix = f"create_{slug}_"
    for name in legacy:
        if name.startswith(preferred_prefix):
            return name
    for name in legacy:
        if category.replace("_", "") in name.replace("_", ""):
            return name
    return legacy[0]


def integration_py(slug: str, category: str, legacy_factory: str) -> str:
    contract_module, contract_class = CATEGORY_CONTRACT[category]
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    label = display_name(slug)
    cat_label = category.replace("_", " ")
    return (
        H
        + f'"""{label} {cat_label} integration (INTEGRATIONS-2D)."""\n\n'
        + "from __future__ import annotations\n\n"
        + "from typing import Protocol, runtime_checkable\n\n"
        + "from pydantic import PrivateAttr\n\n"
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
        + f"    {label} {cat_label} integration.\n\n"
        + f"    The legacy facade ({legacy_factory}) remains separate and backward-compatible.\n"
        + '    """\n\n'
        + f"    config: {prefix}IntegrationConfig = {prefix}IntegrationConfig()\n"
        + f"    _client: {prefix}Client | None = PrivateAttr(default=None)\n\n"
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
    )


def _is_simple_bundle(source: str) -> bool:
    lines = [line.strip() for line in source.splitlines() if line.strip() and not line.strip().startswith("#")]
    non_import = [line for line in lines if not line.startswith("from ") and not line.startswith("import ")]
    return len(non_import) <= 2 and "__all__" in source


def bundle_py(slug: str, category: str, legacy_factory: str, existing: str | None) -> str:
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    contract_factory = contract_factory_name(slug, category)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    label = display_name(slug)

    if existing and _contract_factory_exists(existing, slug, category):
        return existing

    contract_imports = (
        f"from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
        f"from {import_base}.integration import (\n"
        f"    {const},\n"
        f"    {prefix}Integration,\n"
        f"    {prefix}IntegrationConfig,\n"
        f"    {prefix}Client,\n"
        f")\n"
    )
    contract_fn = (
        f"\n\n"
        f"def {contract_factory}(\n"
        f"    *,\n"
        f"    client: {prefix}Client | None = None,\n"
        f"    enabled: bool = False,\n"
        f") -> {prefix}Integration:\n"
        f'    """\n'
        f"    Build a contract-based {label} {category.replace('_', ' ')} integration.\n\n"
        f"    The legacy facade ({legacy_factory}) is unchanged.\n"
        f"    Client must be injected explicitly when enabled=True; disabled by default.\n"
        f'    """\n'
        f"    if enabled and client is None:\n"
        f"        raise IntegrationConfigurationError(\n"
        f'            "{label} {category.replace("_", " ")} integration requires an injected client when enabled=True",\n'
        f"        )\n"
        f"    if client is not None:\n"
        f"        return {prefix}Integration.from_client(client, enabled=enabled)\n"
        f"    return {prefix}Integration.for_provider(\n"
        f"        provider_id={const},\n"
        f'        display_name="{label}",\n'
        f"        config={prefix}IntegrationConfig(enabled=enabled),\n"
        f"    )\n"
    )
    simple_all = (
        f"__all__ = [\n"
        f'    "{legacy_factory}",\n'
        f'    "{contract_factory}",\n'
        f"]\n"
    )

    if existing is None:
        return (
            H
            + f"from intergrax.integrations._shared.p3.factories import {legacy_factory}\n\n"
            + contract_imports
            + "\n"
            + simple_all
            + contract_fn
        )

    if _is_simple_bundle(existing):
        return (
            H
            + f"from intergrax.integrations._shared.p3.factories import {legacy_factory}\n\n"
            + contract_imports
            + "\n"
            + simple_all
            + contract_fn
        )

    updated = existing.rstrip()
    if "IntegrationConfigurationError" not in updated:
        updated += "\n\n" + contract_imports
    updated += contract_fn
    if f'"{contract_factory}"' not in updated and f"'{contract_factory}'" not in updated:
        updated = re.sub(
            r"(__all__\s*=\s*\[)([^\]]*)",
            lambda m: m.group(1) + m.group(2).rstrip() + f'\n    "{contract_factory}",\n',
            updated,
            count=1,
        )
    return updated


def init_py(slug: str, category: str, legacy_factory: str, existing: str | None) -> str:
    prefix = class_prefix(slug, category)
    const = provider_id_const(slug, category)
    contract_factory = contract_factory_name(slug, category)
    import_base = f"intergrax.integrations.providers.{category}.{slug}"

    integration_exports = [
        const,
        f"{prefix}Integration",
        f"{prefix}IntegrationConfig",
        f"{prefix}Client",
    ]
    bundle_exports = [legacy_factory, contract_factory]

    if existing and "export_from_bundle" in existing and const in existing:
        return existing

    if existing and "__getattr__" in existing and len(existing) < 800:
        # Extend simple lazy init
        all_exports = integration_exports + bundle_exports + [f"register_{slug}_integration"]
        lines = [
            H,
            "from intergrax.utils.lazy_export import export_from_bundle\n\n",
            "__all__ = [\n",
        ]
        for item in all_exports:
            lines.append(f'    "{item}",\n')
        lines.append("]\n\n")
        lines.append(f"_BUNDLE_EXPORTS = frozenset(\n    {{\n")
        for item in bundle_exports:
            lines.append(f'        "{item}",\n')
        lines.append("    }\n)\n\n")
        lines.append("_INTEGRATION_EXPORTS = frozenset(\n    {\n")
        for item in integration_exports:
            lines.append(f'        "{item}",\n')
        lines.append("    }\n)\n\n\n")
        lines.append("def __getattr__(name: str):\n")
        lines.append(f'    if name == "register_{slug}_integration":\n')
        lines.append(f"        from {import_base}.register import register_{slug}_integration\n\n")
        lines.append(f"        return register_{slug}_integration\n")
        lines.append("    if name in _BUNDLE_EXPORTS:\n")
        lines.append(f"        from {import_base} import bundle as _bundle\n\n")
        lines.append("        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)\n")
        lines.append("    if name in _INTEGRATION_EXPORTS:\n")
        lines.append(f"        from {import_base} import integration as _integration\n\n")
        lines.append("        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)\n")
        lines.append(f"    raise AttributeError(f\"module {{__name__!r}} has no attribute {{name!r}}\")\n")
        return "".join(lines)

    # Complex init: append contract exports without removing legacy symbols
    if existing and len(existing) >= 800:
        patched = existing
        for symbol in integration_exports + [contract_factory]:
            if symbol not in patched:
                patched = re.sub(
                    r"(__all__\s*=\s*\[)([^\]]*)",
                    lambda m, s=symbol: m.group(1) + m.group(2).rstrip() + f'\n    "{s}",\n',
                    patched,
                    count=1,
                )
        if "_INTEGRATION_EXPORTS" not in patched:
            insert = (
                f"\n_CONTRACT_INTEGRATION_EXPORTS = frozenset(\n    {{\n"
                + "".join(f'        "{s}",\n' for s in integration_exports)
                + "    }\n)\n"
            )
            patched = patched.replace("\ndef __getattr__", insert + "\ndef __getattr__", 1)
            patched = patched.replace(
                "raise AttributeError",
                "    if name in _CONTRACT_INTEGRATION_EXPORTS:\n"
                f"        from {import_base} import integration as _integration\n\n"
                "        return export_from_bundle(_integration, name, _CONTRACT_INTEGRATION_EXPORTS)\n"
                "    raise AttributeError",
                1,
            )
            if "export_from_bundle" not in patched:
                patched = patched.replace(
                    "from __future__ import annotations\n\n",
                    "from __future__ import annotations\n\nfrom intergrax.utils.lazy_export import export_from_bundle\n\n",
                    1,
                )
        if contract_factory not in patched and "_LAZY_EXPORTS" in patched:
            patched = re.sub(
                r"(_LAZY_EXPORTS = frozenset\(\s*\{)([^}]*)(\})",
                lambda m: m.group(1) + m.group(2).rstrip() + f'\n        "{contract_factory}",\n    ' + m.group(3),
                patched,
                count=1,
            )
        return patched

    all_exports = integration_exports + bundle_exports + [f"register_{slug}_integration"]
    lines = [
        H,
        "from intergrax.utils.lazy_export import export_from_bundle\n\n",
        "__all__ = [\n",
    ]
    for item in all_exports:
        lines.append(f'    "{item}",\n')
    lines.append("]\n\n")
    lines.append(f"_BUNDLE_EXPORTS = frozenset(\n    {{\n")
    for item in bundle_exports:
        lines.append(f'        "{item}",\n')
    lines.append("    }\n)\n\n")
    lines.append("_INTEGRATION_EXPORTS = frozenset(\n    {\n")
    for item in integration_exports:
        lines.append(f'        "{item}",\n')
    lines.append("    }\n)\n\n\n")
    lines.append("def __getattr__(name: str):\n")
    lines.append(f'    if name == "register_{slug}_integration":\n')
    lines.append(f"        from {import_base}.register import register_{slug}_integration\n\n")
    lines.append(f"        return register_{slug}_integration\n")
    lines.append("    if name in _BUNDLE_EXPORTS:\n")
    lines.append(f"        from {import_base} import bundle as _bundle\n\n")
    lines.append("        return export_from_bundle(_bundle, name, _BUNDLE_EXPORTS)\n")
    lines.append("    if name in _INTEGRATION_EXPORTS:\n")
    lines.append(f"        from {import_base} import integration as _integration\n\n")
    lines.append("        return export_from_bundle(_integration, name, _INTEGRATION_EXPORTS)\n")
    lines.append(f"    raise AttributeError(f\"module {{__name__!r}} has no attribute {{name!r}}\")\n")
    return "".join(lines)


def usage_md(slug: str, category: str, legacy_factory: str) -> str:
    prefix = class_prefix(slug, category)
    contract_factory = contract_factory_name(slug, category)
    label = display_name(slug)
    return (
        f"# {label} ({slug})\n\n"
        f"Category: `{category}`\n\n"
        "## Legacy facade\n\n"
        f"- `{legacy_factory}()` remains backward-compatible.\n\n"
        "## Contract-based integration\n\n"
        f"- `{prefix}Integration` derives from the category-specific contract.\n"
        f"- Factory: `{contract_factory}()`.\n"
        "- Disabled by default (`enabled=False`).\n"
        "- No vendor SDK or network I/O in the contract adapter.\n"
        "- Injectable `{prefix}Client` required when `enabled=True`.\n\n"
        "## Registry\n\n"
        "- `register.py` remains legacy-compatible.\n"
        "- Registry v2 / contract registry wiring deferred.\n"
    )


def ensure_bundle_from_register(slug: str, category: str, pkg: Path, legacy_factory: str) -> None:
    """Create bundle.py when legacy factory lives only in register.py (yt_dlp)."""
    bundle_path = pkg / "bundle.py"
    if bundle_path.is_file():
        return
    register_path = pkg / "register.py"
    if not register_path.is_file():
        return
    register_src = register_path.read_text(encoding="utf-8")
    if f"def {legacy_factory}" not in register_src:
        return
    import_base = f"intergrax.integrations.providers.{category}.{slug}"
    bundle_src = register_src.split(f"def register_{slug}_integration")[0].rstrip() + "\n"
    if "__all__" not in bundle_src:
        bundle_src += f'\n__all__ = ["{legacy_factory}"]\n'
    bundle_path.write_text(bundle_src, encoding="utf-8")
    register_path.write_text(
        H
        + f'"""Register {slug}."""\n\nfrom __future__ import annotations\n\n'
        + f"from {import_base}.bundle import {legacy_factory}\n"
        + f"from {import_base}.manifest import MANIFEST\n"
        + "from intergrax.integrations.registry.plugin_register import register_from_manifest\n\n\n"
        + f"def register_{slug}_integration(*, override: bool = False) -> None:\n"
        + f"    register_from_manifest(MANIFEST, {legacy_factory}, override=override)\n",
        encoding="utf-8",
    )


def migrate_slug(slug: str, category: str, *, dry_run: bool = False) -> None:
    if slug in DEFERRED_SLUGS:
        return
    if category == "observability_backend":
        return
    pkg = ROOT / "intergrax" / "integrations" / "providers" / category / slug
    if not pkg.is_dir():
        print(f"SKIP missing package: {category}/{slug}")
        return

    legacy_factory = detect_legacy_factory(slug, category, pkg)
    if slug in REGISTER_FACTORY_SLUGS and not dry_run:
        ensure_bundle_from_register(slug, category, pkg, legacy_factory)
    integration_path = pkg / "integration.py"
    bundle_path = pkg / "bundle.py"
    init_path = pkg / "__init__.py"
    usage_path = pkg / "USAGE.md"

    existing_bundle = bundle_path.read_text(encoding="utf-8") if bundle_path.is_file() else None
    existing_init = init_path.read_text(encoding="utf-8") if init_path.is_file() else None

    files = {
        integration_path: integration_py(slug, category, legacy_factory),
        bundle_path: bundle_py(slug, category, legacy_factory, existing_bundle),
        init_path: init_py(slug, category, legacy_factory, existing_init),
        usage_path: usage_md(slug, category, legacy_factory),
    }

    if dry_run:
        print(f"would migrate {category}/{slug} legacy={legacy_factory}")
        return

    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    print(f"migrated {category}/{slug}")


def slugs_for_wave(wave: str) -> list[tuple[str, str]]:
    categories = WAVE_CATEGORIES[wave]
    return sorted(
        (slug, cat)
        for slug, cat in SLUG_CATEGORY.items()
        if cat in categories and cat != "observability_backend" and slug not in DEFERRED_SLUGS
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wave", choices=sorted(WAVE_CATEGORIES), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for slug, category in slugs_for_wave(args.wave):
        migrate_slug(slug, category, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
