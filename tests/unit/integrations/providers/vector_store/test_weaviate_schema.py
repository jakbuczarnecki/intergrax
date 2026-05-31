# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from intergrax.integrations.providers.vector_store.weaviate.schema import (
    metadata_filter_to_weaviate,
)

pytestmark = pytest.mark.unit


class _FakeFilter:
    @staticmethod
    def by_property(name: str) -> "_PropFilter":
        return _PropFilter(name)


class _PropFilter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.ops: List[str] = []

    def equal(self, value: Any) -> "_PropFilter":
        self.ops.append(f"eq:{value}")
        return self

    def contains_any(self, values: Any) -> "_PropFilter":
        self.ops.append(f"any:{values}")
        return self

    def __and__(self, other: "_PropFilter") -> "_CombinedFilter":
        return _CombinedFilter([self, other])


class _CombinedFilter:
    def __init__(self, parts: List[_PropFilter]) -> None:
        self.parts = parts


def test_metadata_filter_to_weaviate_builds_combined_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    import intergrax.integrations.providers.vector_store.weaviate.schema as schema_mod

    class _Query:
        Filter = _FakeFilter

    monkeypatch.setitem(__import__("sys").modules, "weaviate.classes.query", _Query())

    filt = metadata_filter_to_weaviate(
        {"doc_id": "d1", "tenant_id": "lab"},
        default_tenant="lab",
    )
    assert isinstance(filt, _CombinedFilter)
    assert len(filt.parts) == 2
