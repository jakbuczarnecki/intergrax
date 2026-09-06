# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Google Workspace adapter over generic credential resolution (P1.7)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from types import MappingProxyType

from intergrax.integrations.contracts.credential import (
    CredentialRef,
    CredentialResolutionContext,
)
from intergrax.integrations.credentials.errors import sanitize_credential_error_message
from intergrax.integrations.credentials.secrets_store_resolver import (
    SecretsStoreCredentialResolver,
)


class GoogleWorkspaceSecretsStoreCredentialResolver:
    """Resolve Google Workspace credential material at operation boundary."""

    def __init__(
        self,
        *,
        resolver: SecretsStoreCredentialResolver,
        credential_ref: CredentialRef,
        context: CredentialResolutionContext,
    ) -> None:
        self._resolver = resolver
        self._credential_ref = credential_ref
        self._context = context
        self._bound_path = credential_ref.config_credential_ref()

    def resolve_credential(self, credential_ref: str) -> Mapping[str, str]:
        if credential_ref.strip() != self._bound_path:
            raise ValueError("credential reference does not match bound reference")
        resolved = self._resolver.resolve(self._credential_ref, context=self._context)
        try:
            parsed = json.loads(resolved.value)
        except json.JSONDecodeError:
            raise ValueError(
                sanitize_credential_error_message("credential payload is not valid JSON"),
            ) from None
        if not isinstance(parsed, dict) or not parsed:
            raise ValueError(
                sanitize_credential_error_message("credential payload must be a JSON object"),
            )
        material: dict[str, str] = {}
        for key, value in parsed.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(
                    sanitize_credential_error_message(
                        "credential payload keys must be nonblank strings",
                    ),
                )
            if not isinstance(value, str):
                raise ValueError(
                    sanitize_credential_error_message(
                        "credential payload values must be strings",
                    ),
                )
            material[key] = value
        return MappingProxyType(material)
