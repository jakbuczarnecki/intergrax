# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

import pytest

from intergrax.distributed.registry import DistributedProviderRegistry
from intergrax.distributed.contracts.kv_store import DistributedKVStore

pytestmark = pytest.mark.unit

class DummyKVStore(DistributedKVStore):
    def __init__(self, value: int) -> None:
        self.value = value

    async def get(self, key: str) -> bytes | None:  # type: ignore[override]
        raise NotImplementedError

    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> None:  # type: ignore[override]
        raise NotImplementedError

    async def delete(self, key: str) -> None:  # type: ignore[override]
        raise NotImplementedError

    async def compare_and_set(
        self,
        key: str,
        expected: bytes | None,
        value: bytes,
        ttl_seconds: int | None = None,
    ) -> bool:  # type: ignore[override]
        raise NotImplementedError


def test_register_and_get_provider() -> None:
    registry = DistributedProviderRegistry()
    registry.register("dummy", DummyKVStore)

    provider_cls = registry.get_provider("dummy")

    assert provider_cls is DummyKVStore


def test_duplicate_registration_raises() -> None:
    registry = DistributedProviderRegistry()
    registry.register("dummy", DummyKVStore)

    with pytest.raises(ValueError):
        registry.register("dummy", DummyKVStore)


def test_missing_backend_raises() -> None:
    registry = DistributedProviderRegistry()

    with pytest.raises(ValueError):
        registry.get_provider("missing")


def test_create_instantiates_provider() -> None:
    registry = DistributedProviderRegistry()
    registry.register("dummy", DummyKVStore)

    instance = registry.create("dummy", value=42)

    assert isinstance(instance, DummyKVStore)
    assert instance.value == 42