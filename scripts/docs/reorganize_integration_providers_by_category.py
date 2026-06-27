#!/usr/bin/env python3
"""Move integration providers into category subfolders and rewrite import paths."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROVIDERS = ROOT / "intergrax" / "integrations" / "providers"
TESTS = ROOT / "tests" / "unit" / "integrations" / "providers"

# Primary catalog category per slug (matches IntegrationCategory / wire_p2 SPECS).
SLUG_CATEGORY: dict[str, str] = {
    # relational_store
    "sqlite": "relational_store",
    "postgresql": "relational_store",
    "mysql": "relational_store",
    "databricks": "relational_store",
    "oracle": "relational_store",
    "mssql": "relational_store",
    "azure_sql": "relational_store",
    "cloud_sql": "relational_store",
    # document_store
    "cassandra": "document_store",
    "mongodb": "document_store",
    "dynamodb": "document_store",
    # key_value_cache
    "redis": "key_value_cache",
    "memcached": "key_value_cache",
    "elasticache": "key_value_cache",
    # message_bus
    "kafka": "message_bus",
    "celery": "message_bus",
    "rabbitmq": "message_bus",
    "sqs": "message_bus",
    "service_bus": "message_bus",
    "pubsub": "message_bus",
    # object_storage
    "s3": "object_storage",
    "azure_blob": "object_storage",
    "gcs": "object_storage",
    # vector_store
    "pinecone": "vector_store",
    "qdrant": "vector_store",
    "chroma": "vector_store",
    # search_provider
    "google_cse": "search_provider",
    "bing": "search_provider",
    "brave": "search_provider",
    "serpapi": "search_provider",
    # notification_channel (slack/teams also register interaction_surface)
    "slack": "notification_channel",
    "teams": "notification_channel",
    "webhook": "notification_channel",
    "log": "notification_channel",
    "email_smtp": "notification_channel",
    # interaction_surface
    "lab_json": "interaction_surface",
    # collaboration_suite
    "ms365_graph": "collaboration_suite",
    "google_workspace": "collaboration_suite",
    # issue_tracker
    "jira": "issue_tracker",
    "github": "issue_tracker",
    "linear": "issue_tracker",
    "azure_devops": "issue_tracker",
    # wiki_knowledge
    "confluence": "wiki_knowledge",
    "notion": "wiki_knowledge",
    "sharepoint": "wiki_knowledge",
    # observability_backend
    "prometheus": "observability_backend",
    "elasticsearch": "observability_backend",
    "otel": "observability_backend",
    # browser_automation
    "playwright": "browser_automation",
    # cloud_platform
    "aws": "cloud_platform",
    "azure": "cloud_platform",
    "gcp": "cloud_platform",
}

HEADER = (
    "# © Artur Czarnecki. All rights reserved.\n"
    "# Intergrax framework – proprietary and confidential.\n\n"
)


def move_providers() -> None:
    for category in sorted(set(SLUG_CATEGORY.values())):
        (PROVIDERS / category).mkdir(parents=True, exist_ok=True)
        init = PROVIDERS / category / "__init__.py"
        if not init.exists():
            init.write_text(
                HEADER + f'"""Integration providers — ``{category}`` category."""\n',
                encoding="utf-8",
            )

    for slug, category in sorted(SLUG_CATEGORY.items()):
        src = PROVIDERS / slug
        dst = PROVIDERS / category / slug
        if not src.exists():
            if dst.exists():
                continue
            raise FileNotFoundError(f"missing provider dir: {src}")
        if dst.exists():
            continue
        shutil.move(str(src), str(dst))
        print(f"moved {slug} -> {category}/{slug}")


def rewrite_imports() -> None:
    slugs_sorted = sorted(SLUG_CATEGORY.keys(), key=len, reverse=True)
    patterns: list[tuple[re.Pattern[str], str]] = []
    for slug in slugs_sorted:
        category = SLUG_CATEGORY[slug]
        old = f"intergrax.integrations.providers.{slug}"
        new = f"intergrax.integrations.providers.{category}.{slug}"
        patterns.append((re.compile(re.escape(old)), new))
        old_path = f"intergrax/integrations/providers/{slug}/"
        new_path = f"intergrax/integrations/providers/{category}/{slug}/"
        patterns.append((re.compile(re.escape(old_path)), new_path))
        old_path2 = f"integrations/providers/{slug}/"
        new_path2 = f"integrations/providers/{category}/{slug}/"
        patterns.append((re.compile(re.escape(old_path2)), new_path2))

    extensions = {".py", ".md", ".yaml", ".yml", ".toml"}
    skip_dirs = {".git", ".venv", "__pycache__", "build", "node_modules", ".pytest_cache"}
    changed = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in skip_dirs for part in path.parts):
            continue
        if path.suffix not in extensions:
            continue
        if path.name == Path(__file__).name:
            continue
        text = path.read_text(encoding="utf-8")
        original = text
        for pattern, repl in patterns:
            text = pattern.sub(repl, text)
        if text != original:
            path.write_text(text, encoding="utf-8")
            changed += 1
    print(f"rewrote imports in {changed} files")


def move_tests() -> None:
    if not TESTS.exists():
        return
    for test_file in list(TESTS.glob("test_*.py")):
        name = test_file.name
        # test_p2_providers.py stays at category root
        if name == "test_p2_providers.py":
            continue
        slug = name.removeprefix("test_").removesuffix(".py")
        if slug == "qdrant_chroma":
            category = "vector_store"
            dst_dir = TESTS / category
            dst_dir.mkdir(exist_ok=True)
            dst = dst_dir / name
            if not dst.exists() and test_file.exists():
                shutil.move(str(test_file), str(dst))
            continue
        if slug not in SLUG_CATEGORY:
            continue
        category = SLUG_CATEGORY[slug]
        dst_dir = TESTS / category
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / name
        if dst.exists():
            continue
        shutil.move(str(test_file), str(dst))
        print(f"moved test {name} -> {category}/")


def main() -> None:
    move_providers()
    rewrite_imports()
    move_tests()
    print("done")


if __name__ == "__main__":
    main()
