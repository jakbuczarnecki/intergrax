"""Architecture conformance checks for resumable data pack builder core."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ROOT = Path(__file__).resolve().parents[5]
_BUILDER_MODULES = (
    _ROOT
    / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/resumable_builder.py",
    _ROOT
    / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/build_state_machine.py",
    _ROOT
    / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/shard_write.py",
)

_FORBIDDEN_SNIPPETS = (
    "postgresql",
    "qdrant",
    "**updates: object",
    ": object)",
    "from typing import Any",
    "import Any",
)


def test_builder_core_modules_avoid_forbidden_dependencies() -> None:
    for module_path in _BUILDER_MODULES:
        source = module_path.read_text(encoding="utf-8").lower()
        for snippet in _FORBIDDEN_SNIPPETS:
            assert snippet not in source, f"{module_path.name} contains forbidden snippet: {snippet}"
