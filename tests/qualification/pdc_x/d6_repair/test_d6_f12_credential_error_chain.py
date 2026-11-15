# © Artur Czarnecki. All rights reserved.

"""PDC-D6 F-D6-12 — no provider text in the credential-unavailable error chain."""

from __future__ import annotations

import traceback

import pytest
from intergrax.integrations.contracts.credential import (
    CredentialNotFoundError,
    CredentialProviderUnavailableError,
    CredentialRef,
    CredentialResolutionContext,
)
from intergrax.integrations.credentials.secrets_store_resolver import SecretsStoreCredentialResolver
from intergrax.runtime.sandbox.network_egress import canonicalize_network_egress_allowlist

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_PROVIDER_TEXT = "backend failed token=tv-PROVIDER-SECRET-123"


class _Store:
    def __init__(self, error: BaseException | None = None, value: str | None = None) -> None:
        self._error = error
        self._value = value

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        if self._error is not None:
            raise self._error
        assert self._value is not None
        return self._value


def _call(store: _Store) -> None:
    ref = CredentialRef.from_secret_path(provider_id="p", secret_path="secrets/t/p", tenant_id="t")
    context = CredentialResolutionContext(
        tenant_id="t",
        execution_id="e",
        operation="op",
        integration_id="p:i",
        target_scope=canonicalize_network_egress_allowlist(("https://a.example:443",)),
    )
    SecretsStoreCredentialResolver(store).resolve(ref, context=context)  # type: ignore[arg-type]


def test_provider_text_is_absent_from_the_whole_error_chain() -> None:
    with pytest.raises(CredentialProviderUnavailableError) as caught:
        _call(_Store(error=RuntimeError(_PROVIDER_TEXT)))

    error = caught.value
    assert "PROVIDER-SECRET" not in str(error)
    assert error.__cause__ is None
    rendered = "".join(traceback.format_exception(error))
    assert "PROVIDER-SECRET" not in rendered


def test_a_missing_credential_and_an_empty_value_keep_their_typed_errors() -> None:
    with pytest.raises(CredentialNotFoundError):
        _call(_Store(error=KeyError("secrets/t/p")))
    with pytest.raises(CredentialNotFoundError):
        _call(_Store(value="   "))
