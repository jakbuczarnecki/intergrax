# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-M / R5 (F-D6-12): provider text is unreachable from the credential error, ``__context__`` included."""

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


def _chain_text(error: BaseException) -> str:
    return "".join(traceback.format_exception(error)) + repr(error) + str(error)


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (RuntimeError(_PROVIDER_TEXT), CredentialProviderUnavailableError),
        (KeyError(_PROVIDER_TEXT), CredentialNotFoundError),
    ],
)
def test_neither_cause_nor_context_keeps_the_provider_exception(failure: BaseException, expected: type) -> None:
    with pytest.raises(expected) as caught:
        _call(_Store(error=failure))

    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    assert "PROVIDER-SECRET" not in _chain_text(error)


def test_empty_values_keep_their_typed_error() -> None:
    with pytest.raises(CredentialNotFoundError):
        _call(_Store(value="   "))
