# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.tools.providers.platform.contracts import PlatformDeleteSecretInput
from intergrax.tools.providers.platform.service import platform_delete_secret
from intergrax.tools.registry.wiring import ToolWiringContext

pytestmark = pytest.mark.unit


class InMemorySecretsStore:
    def __init__(self) -> None:
        self.values: dict[str, str] = {"apps/demo/key": "secret"}

    def get_secret(self, path: str, *, version: str | None = None) -> str:
        return self.values[path]

    def put_secret(self, path: str, value: str) -> None:
        self.values[path] = value

    def delete_secret(self, path: str) -> None:
        self.values.pop(path, None)


def test_platform_delete_secret_removes_value() -> None:
    store = InMemorySecretsStore()
    ctx = ToolWiringContext(secrets_store=store)
    out = platform_delete_secret(ctx, PlatformDeleteSecretInput(path="apps/demo/key"))
    assert out.deleted is True
    assert "apps/demo/key" not in store.values
