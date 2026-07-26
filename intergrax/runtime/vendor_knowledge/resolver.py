# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""IntegrationProfile-backed resolver for the Vendor Knowledge Facade."""

from __future__ import annotations

from intergrax.integrations.contracts.base import (
    IntegrationCategoryMismatchError,
    IntegrationConfigurationError,
    IntegrationDependencyError,
    UnknownIntegrationError,
)
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.runtime.vendor_knowledge.errors import (
    VendorKnowledgeError,
    VendorKnowledgeErrorCode,
)
from intergrax.runtime.vendor_knowledge.models import KnowledgeSourceRef
from intergrax.utils.attribute_access import optional


class IntegrationProfileVendorResolver:
    """Resolve an existing integration instance from an IntegrationProfile.

    Stateless after construction. Connection-aware resolution is out of scope.
    """

    def __init__(self, *, profile: IntegrationProfile, tenant_id: str) -> None:
        cleaned_tenant = str(tenant_id).strip()
        if not cleaned_tenant:
            raise ValueError("tenant_id must be a non-empty string")
        self._profile = profile
        self._tenant_id = cleaned_tenant

    def resolve(self, *, source: KnowledgeSourceRef) -> object:
        if source.tenant_id != self._tenant_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.TENANT_MISMATCH,
                safe_message="Knowledge source tenant does not match the configured resolver tenant",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        if source.connection_ref is not None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.CONFIGURATION_ERROR,
                safe_message=(
                    "Connection-aware resolution is not configured for this resolver"
                ),
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        try:
            integration = self._profile.resolve(source.integration_kind)
        except VendorKnowledgeError:
            raise
        except (IntegrationConfigurationError, UnknownIntegrationError):
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Requested integration is not available in the configured profile",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except IntegrationCategoryMismatchError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH,
                safe_message="Resolved integration does not match the requested category",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None
        except IntegrationDependencyError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.DEPENDENCY_UNAVAILABLE,
                safe_message="Integration dependency is currently unavailable",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=True,
            ) from None
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Integration resolution returned an unexpected failure",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None

        return self._validate_identity(integration, source=source)

    def _validate_identity(
        self,
        integration: object,
        *,
        source: KnowledgeSourceRef,
    ) -> object:
        try:
            provider_id = optional(integration, "provider_id")
            integration_kind = optional(integration, "integration_kind")
        except AttributeError:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Resolved integration is missing required identity attributes",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None

        if provider_id is None or integration_kind is None:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Resolved integration is missing required identity attributes",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        try:
            resolved_provider = str(provider_id)
            resolved_kind = str(integration_kind)
        except Exception:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INVALID_PROVIDER_RESPONSE,
                safe_message="Resolved integration identity attributes are invalid",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            ) from None

        if resolved_provider != source.provider_id:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_NOT_FOUND,
                safe_message="Resolved integration provider does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        if resolved_kind != source.integration_kind.value:
            raise VendorKnowledgeError(
                code=VendorKnowledgeErrorCode.INTEGRATION_CATEGORY_MISMATCH,
                safe_message="Resolved integration category does not match the requested source",
                provider_id=source.provider_id,
                source_kind=source.source_kind,
                retryable=False,
            )

        return integration
