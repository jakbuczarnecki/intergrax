# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.task_memory.delegation_memory import (
    TaskMemoryMetadataKey,
    apply_delegation_memory_namespace,
)
from intergrax.runtime.task_memory.policy import MemoryAccessPolicy


@pytest.mark.unit
def test_apply_delegation_memory_namespace_merges_allowed() -> None:
    base = MemoryAccessPolicy(allowed_namespaces=frozenset({"task"}))
    updated = apply_delegation_memory_namespace(
        base,
        {TaskMemoryMetadataKey.DELEGATION_MEMORY_NAMESPACE: "t1/delegation/n2"},
    )
    assert updated.allowed_namespaces == frozenset({"task", "t1/delegation/n2"})
