# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.distributed.contracts.kv_store import (
    DistributedKVStore,
    DistributedKVStoreProvider,
)
from tests.unit.queueing.worker.dispatcher_test_kv import DispatcherTestKVStore

pytestmark = pytest.mark.unit


def test_distributed_kv_store_is_not_a_provider() -> None:
    assert not isinstance(DispatcherTestKVStore(), DistributedKVStoreProvider)


def test_distributed_kv_store_provider_matches_kv_store_property() -> None:
    kv = DispatcherTestKVStore()

    class _Provider:
        @property
        def kv_store(self) -> DistributedKVStore:
            return kv

    assert isinstance(_Provider(), DistributedKVStoreProvider)


def test_distributed_kv_store_provider_requires_kv_store_surface() -> None:
    class _MissingKvStore:
        pass

    assert not isinstance(_MissingKvStore(), DistributedKVStoreProvider)
