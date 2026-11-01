# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from testing_support.change_safety.data.loaders import load_evol_x_data_document


def assess_frozen_core_touch(*, changed_paths: frozenset[str]) -> bool:
    registry = load_evol_x_data_document("frozen_core_registry.v1.json")
    surfaces = {row["surface_id"] for row in registry["surfaces"]}
    normalized = {p.replace("\\", "/") for p in changed_paths}
    return bool(normalized & surfaces)
