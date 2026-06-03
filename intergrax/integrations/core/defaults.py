# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Cloud platform default slug mappings — string slugs only (no central enum)."""

from __future__ import annotations

from intergrax.integrations.contracts.base import IntegrationCategory

# When ``IntegrationProfile.cloud_platform`` is set, unset category slots may inherit these slugs.
CLOUD_PLATFORM_DEFAULTS: dict[str, dict[IntegrationCategory, str]] = {
    "aws": {
        IntegrationCategory.OBJECT_STORAGE: "s3",
        IntegrationCategory.MESSAGE_BUS: "sqs",
        IntegrationCategory.DOCUMENT_STORE: "dynamodb",
        IntegrationCategory.KEY_VALUE_CACHE: "elasticache",
    },
    "azure": {
        IntegrationCategory.OBJECT_STORAGE: "azure_blob",
        IntegrationCategory.MESSAGE_BUS: "service_bus",
        IntegrationCategory.RELATIONAL_STORE: "azure_sql",
    },
    "gcp": {
        IntegrationCategory.OBJECT_STORAGE: "gcs",
        IntegrationCategory.MESSAGE_BUS: "pubsub",
        IntegrationCategory.RELATIONAL_STORE: "cloud_sql",
    },
}
