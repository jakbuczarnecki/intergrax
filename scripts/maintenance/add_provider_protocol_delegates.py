#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Add explicit protocol delegation methods to integrations that lost __getattr__."""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS_ROOT = ROOT / "intergrax" / "integrations" / "providers"

SKIP_CATEGORIES = frozenset({"observability_backend", "llm_guardrail"})
DEFERRED_SLUGS = frozenset({"qdrant", "pinecone"})

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


def _protocol_members(protocol: type) -> list[tuple[str, object]]:
    members: list[tuple[str, object]] = []
    for name, member in inspect.getmembers(protocol):
        if name.startswith("_"):
            continue
        if inspect.isfunction(member) or isinstance(member, property):
            members.append((name, member))
    return members


def _method_delegate(name: str, protocol: type) -> str:
    member = getattr(protocol, name, None)
    if member is None or not inspect.isfunction(member):
        return ""
    sig = inspect.signature(member)
    params: list[str] = []
    call_args: list[str] = []
    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue
        if param.default is inspect.Parameter.empty:
            params.append(param_name)
            call_args.append(param_name)
        else:
            default_repr = repr(param.default)
            ann = param.annotation if param.annotation is not inspect.Parameter.empty else ""
            if ann:
                params.append(f"{param_name}: {ann} = {default_repr}")
            else:
                params.append(f"{param_name} = {default_repr}")
            call_args.append(f"{param_name}={param_name}")
    params_str = ", ".join(["self", *params])
    call_str = ", ".join(call_args)
    return f"    def {name}({params_str}):\n        return self._require_client().{name}({call_str})\n\n"


def _property_delegate(name: str) -> str:
    return (
        f"    @property\n"
        f"    def {name}(self):\n"
        f"        return getattr(self._require_client(), {name!r})\n\n"
    )


def _existing_methods(src: str) -> set[str]:
    return set(re.findall(r"\n    def (\w+)\(", src))


def generate_missing_delegates(category: str, src: str) -> str:
    mod_path, cls_name = CATEGORY_PROTOCOL[category]
    mod = importlib.import_module(mod_path)
    protocol = getattr(mod, cls_name)
    existing = _existing_methods(src)
    blocks: list[str] = []
    for name, member in _protocol_members(protocol):
        if name in existing:
            continue
        if isinstance(member, property):
            blocks.append(_property_delegate(name))
        elif inspect.isfunction(member):
            block = _method_delegate(name, protocol)
            if block:
                blocks.append(block)
    return "".join(blocks)


def fix_imports(src: str) -> str:
    if re.search(r"\bAny\b", src) and "from typing import Any" not in src and "import Any" not in src.split("class ", 1)[0]:
        if "from typing import" in src.split("class ", 1)[0]:
            src = re.sub(
                r"(from typing import [^\n]+)\n",
                lambda m: m.group(1) + ", Any\n" if "Any" not in m.group(1) else m.group(0),
                src,
                count=1,
            )
        else:
            src = src.replace(
                "from __future__ import annotations\n\n",
                "from __future__ import annotations\n\nfrom typing import Any\n\n",
                1,
            )
    return src


def patch_integration(category: str, slug: str) -> bool:
    path = PROVIDERS_ROOT / category / slug / "integration.py"
    if not path.is_file():
        return False
    src = path.read_text(encoding="utf-8")
    if "def _require_client" not in src or "_require_inner" in src:
        return False
    delegates = generate_missing_delegates(category, src)
    if not delegates:
        return False
    marker = "\n    def _require_client(self)"
    if marker not in src:
        return False
    src = src.replace(marker, "\n" + delegates + "    def _require_client(self)", 1)
    src = fix_imports(src)
    path.write_text(src, encoding="utf-8")
    return True


def main() -> None:
    patched = 0
    for category_dir in sorted(PROVIDERS_ROOT.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        category = category_dir.name
        if category in SKIP_CATEGORIES or category not in CATEGORY_PROTOCOL:
            continue
        for slug_dir in sorted(category_dir.iterdir()):
            if not slug_dir.is_dir() or slug_dir.name.startswith("_") or slug_dir.name in DEFERRED_SLUGS:
                continue
            if patch_integration(category, slug_dir.name):
                patched += 1
    print(f"patched integrations: {patched}")


if __name__ == "__main__":
    main()
