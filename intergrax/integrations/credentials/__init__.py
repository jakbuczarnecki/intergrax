# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Late-bound credential resolution over existing SecretsStore."""

from intergrax.integrations.credentials.errors import sanitize_credential_error_message
from intergrax.integrations.credentials.google_workspace import (
    GoogleWorkspaceSecretsStoreCredentialResolver,
)
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)

__all__ = [
    "GoogleWorkspaceSecretsStoreCredentialResolver",
    "SecretsStoreCredentialResolver",
    "sanitize_credential_error_message",
]
