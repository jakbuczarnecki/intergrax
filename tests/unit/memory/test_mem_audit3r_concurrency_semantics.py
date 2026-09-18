# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-3-R: explicit reference-store concurrency contract proofs."""

from __future__ import annotations

import inspect

import pytest

from intergrax.memory.stores.in_memory_user_profile_store import InMemoryUserProfileStore
from intergrax.memory.user_profile_store import UserProfileStore

pytestmark = pytest.mark.gate


def test_user_profile_store_protocol_documents_provider_concurrency() -> None:
    doc = UserProfileStore.__doc__ or ""
    lowered = doc.lower()
    assert "concurrency" in lowered
    assert "must document" in lowered


def test_in_memory_user_profile_store_documents_caller_serialized_concurrency() -> None:
    doc = InMemoryUserProfileStore.__doc__ or ""
    lowered = doc.lower()
    assert "not thread-safe" in lowered
    assert "serialized by the caller" in lowered or "serialize" in lowered
    assert "not process-safe" in lowered


def test_in_memory_user_profile_store_has_no_synchronization_primitives() -> None:
    source = inspect.getsource(InMemoryUserProfileStore)
    forbidden = ("threading.Lock", "asyncio.Lock", "RLock", "Semaphore")
    assert not any(token in source for token in forbidden)
