# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative Tier-3 integration selection (Phase M.3) — open catalog manifests."""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.integrations.contracts.base import (
    PROFILE_FIELD_BY_CATEGORY,
    IntegrationCategory,
)
from intergrax.integrations.core.binding import IntegrationBinding
from intergrax.integrations.core.manifest import IntegrationManifest
from intergrax.integrations.core.ref import IntegrationRef, validate_integration_ref
from intergrax.integrations.registry.catalog_manifests import (
    AWS,
    AZURE,
    COHERE_RERANK,
    DOCLING,
    GCP,
    GOOGLE_CSE,
    INMEMORY,
    JINA_RERANK,
    LAB_JSON,
    LANGSMITH,
    LOG,
    OTEL,
    PAGERDUTY,
    QDRANT,
    REDIS,
    SENTRY,
    SQLITE,
)
from intergrax.integrations.core.defaults import CLOUD_PLATFORM_DEFAULTS


class IntegrationProfile(BaseModel):
    """
    Typed provider selection per category for a Tier-3 application.

    Each slot accepts:

    - :class:`~intergrax.integrations.core.manifest.IntegrationManifest` (catalog manifest)
    - :class:`~intergrax.integrations.core.plugin.IntegrationPlugin` subclass (factory via type)
    - pre-built integration **instance** (no catalog factory)
  - slug ``str`` / env (validated against registered catalog)

    Example::

        profile = IntegrationProfile(
            relational_store=SQLITE,
            key_value_cache=REDIS,
            options={SQLITE: {"data_dir": "build/lab"}},
        )
        store = profile.resolve(IntegrationCategory.RELATIONAL_STORE)
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _SLUG_FIELDS: ClassVar[tuple[str, ...]] = tuple(PROFILE_FIELD_BY_CATEGORY.values())

    _BINDING_ACCESSORS: ClassVar[
        dict[str, Callable[["IntegrationProfile"], IntegrationBinding | None]]
    ] = {}

    cloud_platform: IntegrationBinding | None = None
    relational_store: IntegrationBinding | None = None
    document_store: IntegrationBinding | None = None
    key_value_cache: IntegrationBinding | None = None
    message_bus: IntegrationBinding | None = None
    object_storage: IntegrationBinding | None = None
    vector_store: IntegrationBinding | None = None
    search_provider: IntegrationBinding | None = None
    notification_channel: IntegrationBinding | None = None
    interaction_surface: IntegrationBinding | None = None
    collaboration_suite: IntegrationBinding | None = None
    issue_tracker: IntegrationBinding | None = None
    wiki_knowledge: IntegrationBinding | None = None
    observability_backend: IntegrationBinding | None = None
    browser_automation: IntegrationBinding | None = None
    secrets_store: IntegrationBinding | None = None
    graph_store: IntegrationBinding | None = None
    document_parser: IntegrationBinding | None = None
    rerank_provider: IntegrationBinding | None = None
    feature_flag: IntegrationBinding | None = None
    ci_cd: IntegrationBinding | None = None

    options: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _coerce_integration_bindings(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for field_name in cls._SLUG_FIELDS:
            if field_name in normalized:
                normalized[field_name] = validate_integration_ref(
                    field_name,
                    normalized[field_name],
                )
        return normalized

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_option_keys(cls, value: dict[Any, Any] | None) -> dict[str, dict[str, Any]]:
        if not value:
            return {}
        from intergrax.integrations.core.ref import normalize_integration_binding

        normalized: dict[str, dict[str, Any]] = {}
        for key, opts in value.items():
            binding = normalize_integration_binding(key)
            slug = binding.resolved_slug() if binding is not None else str(key).strip().lower()
            if slug:
                normalized[slug] = dict(opts)
        return normalized

    def binding_for_field(self, field_name: str) -> IntegrationBinding | None:
        if field_name not in self._SLUG_FIELDS:
            raise ValueError(f"Unknown integration profile field: {field_name!r}")
        accessor = self._BINDING_ACCESSORS.get(field_name)
        if accessor is None:
            raise ValueError(f"No binding accessor registered for field: {field_name!r}")
        return accessor(self)

    def slug_for_category(self, category: str | IntegrationCategory) -> str | None:
        if isinstance(category, IntegrationCategory):
            category_key = category.value
        else:
            category_key = category.strip().lower()

        field_name = PROFILE_FIELD_BY_CATEGORY.get(category_key)
        if field_name is None or field_name not in self._SLUG_FIELDS:
            return None

        binding = self.binding_for_field(field_name)
        if binding is not None:
            if binding.instance is not None:
                return None
            slug = binding.resolved_slug()
            if slug:
                return slug

        if self.cloud_platform is None:
            return None

        platform_slug = self.cloud_platform.resolved_slug()
        if not platform_slug:
            return None

        defaults = CLOUD_PLATFORM_DEFAULTS.get(platform_slug, {})
        try:
            cat_enum = IntegrationCategory(category_key)
        except ValueError:
            return None
        return defaults.get(cat_enum)

    def options_for_slug(self, slug: IntegrationRef) -> dict[str, Any]:
        from intergrax.integrations.core.ref import normalize_integration_binding

        binding = normalize_integration_binding(slug)
        if binding is None:
            return {}
        key = binding.resolved_slug()
        if not key:
            return {}
        return dict(self.options.get(key, {}))

    def instance_for_category(self, category: IntegrationCategory) -> Any | None:
        field_name = PROFILE_FIELD_BY_CATEGORY.get(category.value)
        if field_name is None:
            return None
        binding = self.binding_for_field(field_name)
        if binding is None:
            return None
        return binding.instance

    def resolve(
        self,
        category: IntegrationCategory,
        *,
        config: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Instantiate the provider for ``category`` using this profile."""
        from intergrax.integrations.registry.factory import resolve_from_profile

        return resolve_from_profile(self, category, config=config)

    @classmethod
    def harness_lab(cls) -> IntegrationProfile:
        return cls(
            relational_store=SQLITE,
            notification_channel=PAGERDUTY,
            observability_backend=SENTRY,
            interaction_surface=LAB_JSON,
            options={
                LANGSMITH.slug: {},
                SENTRY.slug: {},
            },
        )

    @classmethod
    def lab(cls) -> IntegrationProfile:
        return cls(
            relational_store=SQLITE,
            notification_channel=LOG,
            interaction_surface=LAB_JSON,
            document_parser=DOCLING,
        )

    @classmethod
    def harness_environment(cls) -> IntegrationProfile:
        return cls.lab_harness_preset(enable_otel=True)

    @classmethod
    def lab_harness_preset(
        cls,
        *,
        enable_otel: bool = True,
        enable_redis: bool = False,
        enable_qdrant: bool = False,
    ) -> IntegrationProfile:
        options: dict[str, dict[str, Any]] = {}
        if enable_otel:
            options[OTEL.slug] = {}

        return cls(
            relational_store=SQLITE,
            notification_channel=LOG,
            interaction_surface=LAB_JSON,
            document_parser=DOCLING,
            observability_backend=OTEL if enable_otel else None,
            key_value_cache=REDIS if enable_redis else None,
            vector_store=QDRANT if enable_qdrant else None,
            options=options,
        )

    @classmethod
    def legal_product(cls) -> IntegrationProfile:
        return cls(
            relational_store=SQLITE,
            vector_store=INMEMORY,
            document_parser=DOCLING,
            rerank_provider=COHERE_RERANK,
        )

    @classmethod
    def research_product(cls) -> IntegrationProfile:
        return cls(
            relational_store=SQLITE,
            vector_store=INMEMORY,
            document_parser=DOCLING,
            search_provider=GOOGLE_CSE,
            rerank_provider=JINA_RERANK,
        )

    @classmethod
    def with_cloud_platform(cls, platform: IntegrationRef) -> IntegrationProfile:
        binding = validate_integration_ref("cloud_platform", platform)
        return cls(cloud_platform=binding)

    @classmethod
    def lab_stack(cls, *, enable_otel: bool = True) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import lab_stack as _lab_stack

        return _lab_stack(enable_otel=enable_otel)

    @classmethod
    def legal_stack(cls) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import legal_stack as _legal_stack

        return _legal_stack()

    @classmethod
    def research_stack(cls) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import research_stack as _research_stack

        return _research_stack()

    @classmethod
    def data_stack(cls, *, enable_redis: bool = True, enable_qdrant: bool = False) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import data_stack as _data_stack

        return _data_stack(enable_redis=enable_redis, enable_qdrant=enable_qdrant)

    @classmethod
    def observability_stack(cls, *, enable_otel: bool = True) -> IntegrationProfile:
        from intergrax.integrations.registry.presets import observability_stack as _obs_stack

        return _obs_stack(enable_otel=enable_otel)


def default_lab_profile() -> IntegrationProfile:
    return IntegrationProfile.lab()


IntegrationProfile._BINDING_ACCESSORS = {
    "cloud_platform": lambda profile: profile.cloud_platform,
    "relational_store": lambda profile: profile.relational_store,
    "document_store": lambda profile: profile.document_store,
    "key_value_cache": lambda profile: profile.key_value_cache,
    "message_bus": lambda profile: profile.message_bus,
    "object_storage": lambda profile: profile.object_storage,
    "vector_store": lambda profile: profile.vector_store,
    "search_provider": lambda profile: profile.search_provider,
    "notification_channel": lambda profile: profile.notification_channel,
    "interaction_surface": lambda profile: profile.interaction_surface,
    "collaboration_suite": lambda profile: profile.collaboration_suite,
    "issue_tracker": lambda profile: profile.issue_tracker,
    "wiki_knowledge": lambda profile: profile.wiki_knowledge,
    "observability_backend": lambda profile: profile.observability_backend,
    "browser_automation": lambda profile: profile.browser_automation,
    "secrets_store": lambda profile: profile.secrets_store,
    "graph_store": lambda profile: profile.graph_store,
    "document_parser": lambda profile: profile.document_parser,
    "rerank_provider": lambda profile: profile.rerank_provider,
    "feature_flag": lambda profile: profile.feature_flag,
    "ci_cd": lambda profile: profile.ci_cd,
}
