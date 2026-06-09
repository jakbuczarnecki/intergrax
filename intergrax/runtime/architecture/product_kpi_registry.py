# © Artur Czarnecki. All rights reserved.

"""Tenant-scoped product KPI definitions and export (MVP-EVOL.4)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field


class ProductKpiDefinition(BaseModel):
    kpi_id: str
    tenant_id: str
    label: str
    unit: str = "count"
    target_value: float | None = None
    description: str = ""


class ProductKpiObservation(BaseModel):
    observation_id: str
    kpi_id: str
    tenant_id: str
    value: float
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    task_id: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class ProductKpiRegistryStore(BaseModel):
    schema_version: str = "1.0.0"
    definitions: list[ProductKpiDefinition] = Field(default_factory=list)
    observations: list[ProductKpiObservation] = Field(default_factory=list)


class FileProductKpiRegistry:
    def __init__(self, path: Path) -> None:
        self._path = path

    def _load(self) -> ProductKpiRegistryStore:
        if not self._path.is_file():
            return ProductKpiRegistryStore()
        return ProductKpiRegistryStore.model_validate_json(self._path.read_text(encoding="utf-8"))

    def _save(self, store: ProductKpiRegistryStore) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(store.model_dump_json(indent=2), encoding="utf-8")

    def register_definition(self, definition: ProductKpiDefinition) -> None:
        store = self._load()
        store.definitions = [item for item in store.definitions if item.kpi_id != definition.kpi_id]
        store.definitions.append(definition)
        self._save(store)

    def record_observation(self, observation: ProductKpiObservation) -> None:
        store = self._load()
        store.observations.append(observation)
        self._save(store)

    def export_tenant(self, tenant_id: str) -> dict[str, object]:
        store = self._load()
        return {
            "tenant_id": tenant_id,
            "definitions": [
                item.model_dump(mode="json")
                for item in store.definitions
                if item.tenant_id == tenant_id
            ],
            "observations": [
                item.model_dump(mode="json")
                for item in store.observations
                if item.tenant_id == tenant_id
            ],
        }


def default_product_kpi_registry_path() -> Path:
    return Path("build") / "mvp_evolution" / "product_kpi_registry.json"
