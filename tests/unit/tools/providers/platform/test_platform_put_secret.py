# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.providers.platform.contracts import PlatformPutSecretInput
from intergrax.tools.providers.platform.service import platform_put_secret
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemorySecretsStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        return self.values[path]

    def put_secret(self, path: str, value: str) -> None:
        self.values[path] = value

    def delete_secret(self, path: str) -> None:
        self.values.pop(path, None)


def test_platform_put_secret_stores_value() -> None:
    store = InMemorySecretsStore()
    ctx = ToolWiringContext(secrets_store=store)
    out = platform_put_secret(ctx, PlatformPutSecretInput(path=" apps/demo/key ", value="secret"))
    assert out.stored is True
    assert store.values["apps/demo/key"] == "secret"


def test_platform_put_secret_not_configured() -> None:
    with pytest.raises(RuntimeError, match="secrets_store_not_configured"):
        platform_put_secret(ToolWiringContext(), PlatformPutSecretInput(path="x", value="y"))
