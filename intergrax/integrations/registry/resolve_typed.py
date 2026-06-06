# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed helpers for ``IntegrationProfile.resolve`` (optional ergonomics)."""

from __future__ import annotations

from typing import Any, Mapping, Optional, TypeVar, overload

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.document_parser import DocumentParser
from intergrax.integrations.contracts.key_value_cache import KeyValueCache
from intergrax.integrations.contracts.notification_channel import NotificationChannel
from intergrax.integrations.contracts.object_storage import ObjectStorage
from intergrax.integrations.contracts.relational_store import RelationalStore
from intergrax.integrations.contracts.vector_store import VectorStore
from intergrax.integrations.contracts.feature_flag import FeatureFlagBackend
from intergrax.integrations.contracts.ci_cd import CiCdBackend
from intergrax.integrations.registry.profile import IntegrationProfile

T = TypeVar("T")


@overload
def resolve_contract(
    profile: IntegrationProfile,
    category: IntegrationCategory,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> Any: ...


@overload
def resolve_contract(
    profile: IntegrationProfile,
    category: IntegrationCategory,
    *,
    config: Optional[Mapping[str, Any]] = None,
    expected: type[T],
) -> T: ...


def resolve_contract(
    profile: IntegrationProfile,
    category: IntegrationCategory,
    *,
    config: Optional[Mapping[str, Any]] = None,
    expected: type[Any] | None = None,
) -> Any:
    """Resolve and optionally assert instance type for IDE-friendly call sites."""
    value = profile.resolve(category, config=config)
    if expected is not None and not isinstance(value, expected):
        raise TypeError(
            f"Integration for {category.value!r} resolved to {type(value).__name__}, "
            f"expected {expected.__name__}."
        )
    return value


def resolve_relational_store(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> RelationalStore:
    return resolve_contract(
        profile,
        IntegrationCategory.RELATIONAL_STORE,
        config=config,
        expected=RelationalStore,
    )


def resolve_key_value_cache(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> KeyValueCache:
    return resolve_contract(
        profile,
        IntegrationCategory.KEY_VALUE_CACHE,
        config=config,
        expected=KeyValueCache,
    )


def resolve_document_parser(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> DocumentParser:
    return resolve_contract(
        profile,
        IntegrationCategory.DOCUMENT_PARSER,
        config=config,
        expected=DocumentParser,
    )


def resolve_vector_store(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> VectorStore:
    return resolve_contract(
        profile,
        IntegrationCategory.VECTOR_STORE,
        config=config,
        expected=VectorStore,
    )


def resolve_notification_channel(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> NotificationChannel:
    return resolve_contract(
        profile,
        IntegrationCategory.NOTIFICATION_CHANNEL,
        config=config,
        expected=NotificationChannel,
    )


def resolve_object_storage(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> ObjectStorage:
    return resolve_contract(
        profile,
        IntegrationCategory.OBJECT_STORAGE,
        config=config,
        expected=ObjectStorage,
    )


def resolve_feature_flag(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> FeatureFlagBackend:
    return resolve_contract(
        profile,
        IntegrationCategory.FEATURE_FLAG,
        config=config,
        expected=FeatureFlagBackend,
    )


def resolve_ci_cd(
    profile: IntegrationProfile,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> CiCdBackend:
    return resolve_contract(
        profile,
        IntegrationCategory.CI_CD,
        config=config,
        expected=CiCdBackend,
    )
