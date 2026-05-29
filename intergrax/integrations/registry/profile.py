# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Declarative Tier-3 integration selection (Phase M.3)."""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.integrations.contracts.base import (
    PROFILE_FIELD_BY_CATEGORY,
    IntegrationCategory,
)
from intergrax.integrations.registry.slugs import (
    CLOUD_PLATFORM_DEFAULTS,
    FIELD_SLUGS,
    IntegrationSlug,
    SlugInput,
    validate_field_slug,
)


class IntegrationProfile(BaseModel):
    """
    Typed provider selection per category for a Tier-3 application.

    Use ``IntegrationSlug`` members — not raw strings — in application code::

        IntegrationProfile(
            relational_store=IntegrationSlug.SQLITE,
            key_value_cache=IntegrationSlug.REDIS,
            cloud_platform=IntegrationSlug.AWS,
        )

    Or a preset::

        IntegrationProfile.lab()
        IntegrationProfile.with_cloud_platform(IntegrationSlug.AWS)

    Env/YAML may still supply strings; they are coerced to ``IntegrationSlug``.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    _SLUG_FIELDS: ClassVar[tuple[str, ...]] = tuple(FIELD_SLUGS.keys())

    cloud_platform: IntegrationSlug | None = None
    relational_store: IntegrationSlug | None = None
    document_store: IntegrationSlug | None = None
    key_value_cache: IntegrationSlug | None = None
    message_bus: IntegrationSlug | None = None
    object_storage: IntegrationSlug | None = None
    vector_store: IntegrationSlug | None = None
    search_provider: IntegrationSlug | None = None
    notification_channel: IntegrationSlug | None = None
    interaction_surface: IntegrationSlug | None = None
    collaboration_suite: IntegrationSlug | None = None
    issue_tracker: IntegrationSlug | None = None
    wiki_knowledge: IntegrationSlug | None = None
    observability_backend: IntegrationSlug | None = None
    browser_automation: IntegrationSlug | None = None

    options: dict[IntegrationSlug, dict[str, Any]] = Field(default_factory=dict)

    @field_validator(*_SLUG_FIELDS, mode="before")
    @classmethod
    def _coerce_and_validate_slug(cls, value: SlugInput | None, info) -> IntegrationSlug | None:
        if value is None or value == "":
            return None
        field_name = info.field_name
        assert field_name is not None
        return validate_field_slug(field_name, value)

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_option_keys(cls, value: dict[Any, Any] | None) -> dict[IntegrationSlug, dict[str, Any]]:
        if not value:
            return {}
        from intergrax.integrations.registry.slugs import coerce_slug

        return {coerce_slug(key): dict(opts) for key, opts in value.items()}

    def slug_for_category(self, category: str | IntegrationCategory) -> str | None:
        if isinstance(category, IntegrationCategory):
            category_key = category.value
        else:
            category_key = category.strip().lower()

        field_name = PROFILE_FIELD_BY_CATEGORY.get(category_key)
        if field_name is None:
            return None

        explicit = getattr(self, field_name, None)
        if explicit is not None:
            return explicit.value

        if self.cloud_platform is None:
            return None

        defaults = CLOUD_PLATFORM_DEFAULTS.get(self.cloud_platform, {})
        try:
            cat_enum = IntegrationCategory(category_key)
        except ValueError:
            return None
        default_slug = defaults.get(cat_enum)
        return default_slug.value if default_slug is not None else None

    def options_for_slug(self, slug: SlugInput) -> dict[str, Any]:
        from intergrax.integrations.registry.slugs import coerce_slug

        return dict(self.options.get(coerce_slug(slug), {}))

    @classmethod
    def lab(cls) -> IntegrationProfile:
        """Laboratory defaults — no external vendors required."""
        return cls(
            relational_store=IntegrationSlug.SQLITE,
            notification_channel=IntegrationSlug.LOG,
            interaction_surface=IntegrationSlug.LAB_JSON,
        )

    @classmethod
    def with_cloud_platform(cls, platform: IntegrationSlug) -> IntegrationProfile:
        """Select a cloud facade; unset infra categories inherit platform defaults on resolve."""
        validate_field_slug("cloud_platform", platform)
        return cls(cloud_platform=platform)


def default_lab_profile() -> IntegrationProfile:
    """Alias for ``IntegrationProfile.lab()``."""
    return IntegrationProfile.lab()
