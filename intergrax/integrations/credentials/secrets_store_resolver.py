# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SecretsStore-backed credential resolver (P1.7)."""

from __future__ import annotations

from intergrax.integrations.contracts.credential import (
    CredentialNotFoundError,
    CredentialProviderUnavailableError,
    CredentialRef,
    CredentialResolutionContext,
    CredentialScopeMismatchError,
    ResolvedCredential,
)
from intergrax.integrations.contracts.secrets_store import SecretsStore
from intergrax.integrations.credentials.errors import sanitize_credential_error_message


class SecretsStoreCredentialResolver:
    """Resolve :class:`CredentialRef` through the canonical :class:`SecretsStore` seam."""

    def __init__(self, secrets_store: SecretsStore) -> None:
        self._secrets_store = secrets_store

    def resolve(
        self,
        ref: CredentialRef,
        *,
        context: CredentialResolutionContext,
    ) -> ResolvedCredential:
        self._assert_tenant_scope(ref, context)
        # The typed errors are raised outside the ``except`` blocks: ``raise ... from None`` only hides the
        # provider exception from the rendered traceback, ``__context__`` would still hold its text (F-D6-12).
        value: object = None
        failure: str | None = None
        try:
            value = self._secrets_store.get_secret(ref.secret_path, version=ref.version)
        except KeyError:
            failure = "not_found"
        except CredentialScopeMismatchError:
            raise
        except Exception:
            failure = "unavailable"
        if failure == "unavailable":
            raise CredentialProviderUnavailableError(
                sanitize_credential_error_message(
                    f"credential provider unavailable for provider_id={ref.provider_id}",
                ),
            )
        if failure == "not_found" or not isinstance(value, str) or not value.strip():
            raise CredentialNotFoundError(
                sanitize_credential_error_message(
                    f"credential not found for provider_id={ref.provider_id}",
                ),
            )
        return ResolvedCredential(
            ref=ref,
            value=value,
            resolved_version=ref.version,
        )

    @staticmethod
    def _assert_tenant_scope(
        ref: CredentialRef,
        context: CredentialResolutionContext,
    ) -> None:
        context_tenant = context.tenant_id.strip()
        if not context_tenant:
            raise CredentialScopeMismatchError("credential resolution requires tenant context")
        if ref.tenant_id is None:
            return
        if ref.tenant_id.strip() != context_tenant:
            raise CredentialScopeMismatchError(
                "credential reference tenant does not match resolution context",
            )
