# © Artur Czarnecki. All rights reserved.
"""Fix runtime delegation on cutover integration.py files."""

from __future__ import annotations

import re
from pathlib import Path

from intergrax.integrations.providers.layout import SLUG_CATEGORY

REQUIRE_RUNTIME = """    def _require_runtime(self) -> Any:
        private = object.__getattribute__(self, "__pydantic_private__")
        runtime = private.get("_runtime")
        if runtime is None:
            runtime = private.get("_backend")
        if runtime is None:
            runtime = private.get("_inner")
        if runtime is None:
            raise IntegrationConfigurationError(
                f"{type(self).__name__} requires a runtime delegate for catalog operations",
            )
        return runtime
"""

GETATTR = """
    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            private = object.__getattribute__(self, "__pydantic_private__")
            if name in private:
                return private[name]
            raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")
        return getattr(self._require_runtime(), name)
"""

OLD_GETATTR = re.compile(
    r"\n    def __getattr__\(self, name: str\) -> object:\n"
    r"        if name\.startswith\(\"_\"\):\n"
    r"            raise AttributeError\(f\"\{type\(self\)\.__name__\!r\} object has no attribute \{name\!r\}\"\)\n"
    r"        return getattr\(self\._require_runtime\(\), name\)\n",
    re.MULTILINE,
)

OLD_REQUIRE = re.compile(
    r"    def _require_runtime\(self\) -> Any:.*?return runtime\n",
    re.MULTILINE | re.DOTALL,
)

NEED_GETATTR_CATEGORIES = {
    "document_store",
    "key_value_cache",
    "interaction_surface",
    "collaboration_suite",
    "wiki_knowledge",
    "cloud_platform",
    "document_parser",
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
    "message_bus",
    "issue_tracker",
    "browser_automation",
    "secrets_store",
    "graph_store",
    "notification_channel",
    "observability_backend",
    "rerank_provider",
}

DEFERRED = {
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


def main() -> None:
    updated_req = 0
    updated_get = 0
    added_get = 0
    for slug, category in SLUG_CATEGORY.items():
        if category == "llm_guardrail" or slug in DEFERRED:
            continue
        path = Path("intergrax/integrations/providers") / category / slug / "integration.py"
        if not path.is_file():
            continue
        src = path.read_text(encoding="utf-8")
        changed = False

        if "def _require_runtime" in src:
            new_src, count = OLD_REQUIRE.subn(REQUIRE_RUNTIME, src, count=1)
            if count:
                src = new_src
                updated_req += 1
                changed = True

        if OLD_GETATTR.search(src):
            src = OLD_GETATTR.sub(GETATTR, src, count=1)
            updated_get += 1
            changed = True
        elif category in NEED_GETATTR_CATEGORIES and "def __getattr__" not in src:
            if ".register(" in src:
                src = re.sub(r"\n(\w+\.register\()", GETATTR + r"\n\1", src, count=1)
            else:
                src = src.rstrip() + GETATTR + "\n"
            added_get += 1
            changed = True

        if changed:
            path.write_text(src, encoding="utf-8")

    print(f"updated _require_runtime: {updated_req}, updated __getattr__: {updated_get}, added __getattr__: {added_get}")


if __name__ == "__main__":
    main()
