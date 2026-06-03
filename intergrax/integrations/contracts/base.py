# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shared integration catalog types (§7.1.1, Phase M.1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Sequence


class IntegrationStatus(str, Enum):
    STABLE = "stable"
    BETA = "beta"
    DEPRECATED = "deprecated"


class IntegrationCategory(str, Enum):
    RELATIONAL_STORE = "relational_store"
    DOCUMENT_STORE = "document_store"
    KEY_VALUE_CACHE = "key_value_cache"
    MESSAGE_BUS = "message_bus"
    OBJECT_STORAGE = "object_storage"
    VECTOR_STORE = "vector_store"
    SEARCH_PROVIDER = "search_provider"
    NOTIFICATION_CHANNEL = "notification_channel"
    INTERACTION_SURFACE = "interaction_surface"
    COLLABORATION_SUITE = "collaboration_suite"
    ISSUE_TRACKER = "issue_tracker"
    WIKI_KNOWLEDGE = "wiki_knowledge"
    OBSERVABILITY_BACKEND = "observability_backend"
    BROWSER_AUTOMATION = "browser_automation"
    CLOUD_PLATFORM = "cloud_platform"
    SECRETS_STORE = "secrets_store"
    GRAPH_STORE = "graph_store"
    DOCUMENT_PARSER = "document_parser"
    RERANK_PROVIDER = "rerank_provider"


@dataclass(frozen=True)
class IntegrationMetadata:
    slug: str
    categories: tuple[IntegrationCategory, ...]
    status: IntegrationStatus = IntegrationStatus.STABLE
    env_prefix: str = ""


IntegrationFactory = Callable[..., Any]


@dataclass(frozen=True)
class IntegrationEntry:
    """Catalog row: slug → lazy factory for one or more category contracts."""

    slug: str
    categories: tuple[IntegrationCategory, ...]
    factory: IntegrationFactory
    status: IntegrationStatus = IntegrationStatus.STABLE
    env_prefix: str = ""
    description: str = ""

    @property
    def metadata(self) -> IntegrationMetadata:
        return IntegrationMetadata(
            slug=self.slug,
            categories=self.categories,
            status=self.status,
            env_prefix=self.env_prefix,
        )


@dataclass(frozen=True)
class HealthStatus:
    slug: str
    healthy: bool
    detail: str = ""


class IntegrationError(Exception):
    """Base error for integration catalog resolution and wiring."""


class IntegrationDependencyError(IntegrationError):
    """Backend unavailable, circuit open, or dependency timeout."""

    def __init__(self, message: str, *, integration_name: str = "") -> None:
        super().__init__(message)
        self.integration_name = integration_name


class IntegrationConfigurationError(IntegrationError):
    """Profile/env does not specify a slug for a requested category."""


class IntegrationCategoryMismatchError(IntegrationError):
    def __init__(self, slug: str, category: str) -> None:
        super().__init__(
            f"Integration '{slug}' is not registered for category '{category}'."
        )
        self.slug = slug
        self.category = category


class UnknownIntegrationError(IntegrationError):
    def __init__(self, slug: str) -> None:
        super().__init__(f"Integration slug '{slug}' is not registered in the catalog.")
        self.slug = slug


class UnknownIntegrationCategoryError(IntegrationError):
    def __init__(self, category: str) -> None:
        super().__init__(f"Unknown integration category '{category}'.")
        self.category = category


# Cloud facade defaults live in ``registry.slugs.CLOUD_PLATFORM_DEFAULTS`` (typed slugs).


PROFILE_FIELD_BY_CATEGORY: dict[str, str] = {
    IntegrationCategory.RELATIONAL_STORE.value: "relational_store",
    IntegrationCategory.DOCUMENT_STORE.value: "document_store",
    IntegrationCategory.KEY_VALUE_CACHE.value: "key_value_cache",
    IntegrationCategory.MESSAGE_BUS.value: "message_bus",
    IntegrationCategory.OBJECT_STORAGE.value: "object_storage",
    IntegrationCategory.VECTOR_STORE.value: "vector_store",
    IntegrationCategory.SEARCH_PROVIDER.value: "search_provider",
    IntegrationCategory.NOTIFICATION_CHANNEL.value: "notification_channel",
    IntegrationCategory.INTERACTION_SURFACE.value: "interaction_surface",
    IntegrationCategory.COLLABORATION_SUITE.value: "collaboration_suite",
    IntegrationCategory.ISSUE_TRACKER.value: "issue_tracker",
    IntegrationCategory.WIKI_KNOWLEDGE.value: "wiki_knowledge",
    IntegrationCategory.OBSERVABILITY_BACKEND.value: "observability_backend",
    IntegrationCategory.BROWSER_AUTOMATION.value: "browser_automation",
    IntegrationCategory.CLOUD_PLATFORM.value: "cloud_platform",
    IntegrationCategory.SECRETS_STORE.value: "secrets_store",
    IntegrationCategory.GRAPH_STORE.value: "graph_store",
    IntegrationCategory.DOCUMENT_PARSER.value: "document_parser",
    IntegrationCategory.RERANK_PROVIDER.value: "rerank_provider",
}


def normalize_category(category: str | IntegrationCategory) -> IntegrationCategory:
    if isinstance(category, IntegrationCategory):
        return category
    raw = category.strip().lower()
    try:
        return IntegrationCategory(raw)
    except ValueError as exc:
        raise UnknownIntegrationCategoryError(raw) from exc


def categories_for_profile_field(field_name: str) -> Sequence[IntegrationCategory]:
    matched = [
        IntegrationCategory(key)
        for key, value in PROFILE_FIELD_BY_CATEGORY.items()
        if value == field_name
    ]
    return matched
