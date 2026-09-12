"""Load and validate vendor-neutral dataset package for PostgreSQL materialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning.reference.dataset_manifest import (
    DatasetManifestResolution,
    InvalidDatasetError,
    resolve_variant_from_manifest,
)


@dataclass(frozen=True, slots=True)
class LoadedScenarioPackage:
    resolution: DatasetManifestResolution
    manifest: dict[str, Any]
    shared: dict[str, dict[str, Any]]
    variant_document: dict[str, Any]


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InvalidDatasetError(f"invalid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise InvalidDatasetError(f"JSON root must be an object: {path}")
    return document


def _load_shared_entities(
    dataset_root: Path,
    manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    shared_entities = manifest.get("shared_entities")
    if not isinstance(shared_entities, dict):
        raise InvalidDatasetError("manifest shared_entities must be an object")
    loaded: dict[str, dict[str, Any]] = {}
    for key, relative in shared_entities.items():
        if not isinstance(relative, str) or not relative:
            raise InvalidDatasetError(f"shared entity path invalid for {key!r}")
        path = dataset_root / relative
        if not path.is_file():
            raise InvalidDatasetError(f"shared entity missing: {path}")
        loaded[key] = _read_json_object(path)
    return loaded


def _require_shared_entity(shared: dict[str, dict[str, Any]], key: str) -> dict[str, Any]:
    entity = shared.get(key)
    if not isinstance(entity, dict):
        raise InvalidDatasetError(f"shared entity {key} must be present and an object")
    return entity


def _validate_shared_graph(shared: dict[str, dict[str, Any]]) -> None:
    order = _require_shared_entity(shared, "order")
    effect = _require_shared_entity(shared, "external_effect")
    knowledge = _require_shared_entity(shared, "application_knowledge_at_entry")
    inventory = _require_shared_entity(shared, "inventory_context")
    communication = _require_shared_entity(shared, "communication_event")

    order_id = order.get("order_id")
    effect_order = effect.get("related_order_id")
    knowledge_order = knowledge.get("related_order_id")
    inventory_order = inventory.get("related_order_id")
    if not isinstance(order_id, str) or not order_id:
        raise InvalidDatasetError("order.order_id must be a non-empty string")
    if {effect_order, knowledge_order, inventory_order} != {order_id}:
        raise InvalidDatasetError("shared entity order_id correlation mismatch")

    effect_id = effect.get("effect_id")
    knowledge_effect = knowledge.get("related_effect_id")
    communication_effect = communication.get("related_effect_id")
    if not isinstance(effect_id, str) or not effect_id:
        raise InvalidDatasetError("external_effect.effect_id must be a non-empty string")
    if {knowledge_effect, communication_effect} != {effect_id}:
        raise InvalidDatasetError("shared entity effect_id correlation mismatch")


def _validate_variant_refs(
    variant_document: dict[str, Any],
    shared: dict[str, dict[str, Any]],
) -> None:
    refs = variant_document.get("shared_entity_refs")
    if not isinstance(refs, dict):
        raise InvalidDatasetError("variant shared_entity_refs must be an object")
    order = shared["order"]
    effect = shared["external_effect"]
    inventory = shared["inventory_context"]
    communication = shared["communication_event"]
    expected = {
        "order_id": order["order_id"],
        "effect_id": effect["effect_id"],
        "reservation_id": inventory["reservation_id"],
        "communication_event_id": communication["communication_event_id"],
    }
    for key, expected_value in expected.items():
        actual = refs.get(key)
        if actual != expected_value:
            raise InvalidDatasetError(
                f"variant shared_entity_refs.{key} mismatch: expected {expected_value!r}, got {actual!r}"
            )


def load_scenario_package(
    *,
    dataset_package_root: Path,
    qualification_id: str,
    scenario_slug: str,
    variant_id: str,
) -> LoadedScenarioPackage:
    resolution = resolve_variant_from_manifest(
        dataset_package_root=dataset_package_root,
        qualification_id=qualification_id,
        scenario_slug=scenario_slug,
        variant_id=variant_id,
    )
    manifest = _read_json_object(dataset_package_root / "manifest.json")
    shared = _load_shared_entities(dataset_package_root, manifest)
    _validate_shared_graph(shared)
    variant_path = dataset_package_root / resolution.variant.relative_path
    variant_document = _read_json_object(variant_path)
    _validate_variant_refs(variant_document, shared)
    if variant_document.get("external_reality") is None:
        raise InvalidDatasetError("variant slice must include external_reality")
    if variant_document.get("reconciliation") is None:
        raise InvalidDatasetError("variant slice must include reconciliation")
    return LoadedScenarioPackage(
        resolution=resolution,
        manifest=manifest,
        shared=shared,
        variant_document=variant_document,
    )
