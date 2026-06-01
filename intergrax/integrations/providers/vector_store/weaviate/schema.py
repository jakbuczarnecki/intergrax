# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Weaviate schema migration and multi-tenant collection setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1
SCHEMA_VERSION_PROPERTY = "intergrax_schema_version"


@dataclass(frozen=True)
class WeaviateSchemaConfig:
    collection_name: str
    schema_version: int = SCHEMA_VERSION
    multi_tenant: bool = True
    tenant_id: str = "default"


def ensure_weaviate_collection(
    client: Any,
    cfg: WeaviateSchemaConfig,
) -> Any:
    """
    Create or migrate a Weaviate collection for RAG.

    - Enables native multi-tenancy when ``cfg.multi_tenant`` is true.
    - Stores ``intergrax_schema_version`` on collection config for migration checks.
    """
    from weaviate.classes.config import Configure, DataType, Property

    collection_name = cfg.collection_name
    exists = client.collections.exists(collection_name)
    if not exists:
        multi_tenancy = Configure.multi_tenancy(enabled=cfg.multi_tenant) if cfg.multi_tenant else None
        create_kwargs: Dict[str, Any] = {
            "name": collection_name,
            "properties": _base_properties(),
            "vectorizer_config": Configure.Vectorizer.none(),
        }
        if multi_tenancy is not None:
            create_kwargs["multi_tenancy_config"] = multi_tenancy
        client.collections.create(**create_kwargs)
    collection = client.collections.get(collection_name)
    _ensure_tenant(collection, cfg)
    _migrate_if_needed(collection, cfg)
    return collection


def _base_properties() -> List[Any]:
    from weaviate.classes.config import DataType, Property

    return [
        Property(name="text", data_type=DataType.TEXT),
        Property(name="tenant_id", data_type=DataType.TEXT),
        Property(name="doc_id", data_type=DataType.TEXT),
        Property(name=SCHEMA_VERSION_PROPERTY, data_type=DataType.INT),
    ]


def _ensure_tenant(collection: Any, cfg: WeaviateSchemaConfig) -> None:
    if not cfg.multi_tenant:
        return
    tenants = collection.tenants.get()
    tenant_names = {getattr(t, "name", str(t)) for t in (tenants or [])}
    if cfg.tenant_id not in tenant_names:
        collection.tenants.create([cfg.tenant_id])


def _migrate_if_needed(collection: Any, cfg: WeaviateSchemaConfig) -> None:
    """Best-effort schema migration — add missing properties for older collections."""
    try:
        config = collection.config.get()
        prop_names = {p.name for p in (config.properties or [])}
        from weaviate.classes.config import DataType, Property

        missing = []
        for prop in _base_properties():
            if prop.name not in prop_names:
                missing.append(prop)
        for prop in missing:
            collection.config.add_property(prop)
    except Exception:
        return


def metadata_filter_to_weaviate(
    conditions: Dict[str, Any],
    *,
    default_tenant: str,
) -> Optional[Any]:
    """Translate ``MetadataFilter.conditions`` into a Weaviate Filter object."""
    if not conditions:
        return None
    try:
        from weaviate.classes.query import Filter
    except ImportError:
        return None

    filters = []
    for key, value in conditions.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            filters.append(Filter.by_property(key).contains_any(list(value)))
        elif isinstance(value, dict):
            op = str(value.get("op", "eq")).lower()
            val = value.get("value")
            if op in ("eq", "equal"):
                filters.append(Filter.by_property(key).equal(val))
            elif op in ("neq", "not_equal"):
                filters.append(Filter.by_property(key).not_equal(val))
            elif op == "like":
                filters.append(Filter.by_property(key).like(str(val)))
            elif op == "gt":
                filters.append(Filter.by_property(key).greater_than(val))
            elif op == "gte":
                filters.append(Filter.by_property(key).greater_or_equal(val))
            elif op == "lt":
                filters.append(Filter.by_property(key).less_than(val))
            elif op == "lte":
                filters.append(Filter.by_property(key).less_or_equal(val))
        else:
            filters.append(Filter.by_property(key).equal(value))

    if "tenant_id" not in conditions:
        filters.append(Filter.by_property("tenant_id").equal(default_tenant))

    if not filters:
        return None
    combined = filters[0]
    for extra in filters[1:]:
        combined = combined & extra
    return combined
