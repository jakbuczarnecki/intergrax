#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Inline observability legacy backend delegation into integration.py."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBS_ROOT = ROOT / "intergrax" / "integrations" / "providers" / "observability_backend"

REST_CLIENT_SLUGS = frozenset({"braintrust", "langsmith", "prometheus", "elasticsearch", "opensearch"})

REQUIRE_RUNTIME_BLOCK = re.compile(
    r"\n    def _require_runtime\(self\) -> Any:.*?        return runtime\n",
    re.MULTILINE | re.DOTALL,
)

GETATTR_BLOCK = re.compile(
    r"\n    def __getattr__\(self, name: str\) -> object:.*?        return getattr\(self\._require_runtime\(\), name\)\n",
    re.MULTILINE | re.DOTALL,
)

CATALOG_IMPORT = (
    "from intergrax.integrations.providers.observability_backend._catalog_client import (\n"
    "    ObservabilityCatalogClient,\n"
    "    require_observability_catalog_client,\n"
    ")\n"
)


def _client_type(slug: str) -> str:
    if slug in REST_CLIENT_SLUGS:
        if slug == "langsmith":
            return "LangSmithRestClient"
        return f"{slug.capitalize()}RestClient" if slug != "elasticsearch" else "ElasticsearchRestClient"
    return "ObservabilityCatalogClient"


def _client_import(slug: str) -> str:
    if slug not in REST_CLIENT_SLUGS:
        return CATALOG_IMPORT
    if slug == "langsmith":
        cls = "LangSmithRestClient"
    elif slug == "elasticsearch":
        cls = "ElasticsearchRestClient"
    elif slug == "opensearch":
        cls = "OpenSearchRestClient"
    else:
        cls = f"{slug.capitalize()}RestClient"
    return (
        f"from intergrax.integrations.providers.observability_backend.{slug}.client import {cls}\n"
    )


def _require_client_block(client_type: str) -> str:
    if client_type == "ObservabilityCatalogClient":
        return """
    def _require_client(self) -> ObservabilityCatalogClient:
        return require_observability_catalog_client(self, self._client)
"""
    return f"""
    def _require_client(self) -> {client_type}:
        if self._client is None:
            raise IntegrationConfigurationError(
                f"{{type(self).__name__}} requires a catalog client for query operations",
            )
        return self._client
"""


def migrate_integration(slug: str) -> bool:
    path = OBS_ROOT / slug / "integration.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    if "def _require_runtime" not in src:
        return False

    client_type = _client_type(slug)
    client_import = _client_import(slug)

    src = src.replace(
        "    _backend: Any | None = PrivateAttr(default=None)",
        f"    _client: {client_type} | None = PrivateAttr(default=None)",
    )
    src = src.replace("from_backend", "from_client")
    src = src.replace("def from_client(\n        cls,\n        backend: Any,", f"def from_client(\n        cls,\n        client: {client_type},")
    src = src.replace("integration._backend = backend", "integration._client = client")
    src = src.replace("via from_client().", "via from_client().")
    src = src.replace("delegates to this class via from_client().", "owns catalog query behavior; legacy factories use from_client().")

    src = re.sub(
        r"\n    @property\n    def backend\(self\) -> Any \| None:\n        return self\._backend\n",
        "\n    @property\n    def client(self) -> "
        + client_type
        + " | None:\n        return self._client\n",
        src,
        count=1,
    )

    src = src.replace("_require_runtime()", "_require_client()")
    src = REQUIRE_RUNTIME_BLOCK.sub(_require_client_block(client_type), src, count=1)
    src = GETATTR_BLOCK.sub("\n", src, count=1)

    if client_import.strip() not in src:
        marker = "from intergrax.integrations.contracts.base import IntegrationConfigurationError\n"
        src = src.replace(marker, marker + client_import)

    if "from typing import Any, Protocol" in src and "Any" not in src.split("class ", 1)[0]:
        src = src.replace("from typing import Any, Protocol", "from typing import Protocol")

    path.write_text(src, encoding="utf-8")
    return True


def migrate_shell_bundle(slug: str) -> bool:
    path = OBS_ROOT / slug / "bundle.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    if "from_backend" not in src:
        return False
    src = src.replace(".from_backend(runtime)", ".from_client(runtime)")
    path.write_text(src, encoding="utf-8")
    return True


def main() -> None:
    migrated = 0
    bundles = 0
    for slug_dir in sorted(OBS_ROOT.iterdir()):
        if not slug_dir.is_dir() or slug_dir.name.startswith("_"):
            continue
        slug = slug_dir.name
        if migrate_integration(slug):
            migrated += 1
        if migrate_shell_bundle(slug):
            bundles += 1
    print(f"migrated integrations: {migrated}, bundles: {bundles}")


if __name__ == "__main__":
    main()
