# © Artur Czarnecki. All rights reserved.

"""Typed Memory store plugin discovery (ENTERPRISE-5 / BLOCK D)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.core.plugins.admission import (
    DomainPluginLoadReport,
    PluginAdmissionReasonCode,
    PluginAdmissionRejection,
)
from intergrax.core.plugins.discovery import (
    EP_MEMORY_STORES,
    EntryPointLoadResult,
    EntryPointSpec,
    LoadIsolation,
    load_entry_point_targets,
    resolve_entry_point_plugin_type,
)
from intergrax.core.plugins.errors import PluginLoadError
from intergrax.memory.resolver.classifier import (
    ClassifiedMemoryStorePlugin,
    classify_memory_store_plugin,
    classify_memory_store_plugin_record,
)
from intergrax.memory.resolver.errors import MemoryStorePluginResolutionError


@dataclass(frozen=True, slots=True)
class MemoryStorePluginDiscoveryResult:
    """Memory store plugin candidates plus canonical EP bootstrap evidence."""

    plugins: tuple[ClassifiedMemoryStorePlugin, ...]
    load_report: DomainPluginLoadReport


@dataclass(frozen=True, slots=True)
class MemoryStorePluginCatalog:
    """Indexed Memory store plugin candidates with EP bootstrap evidence."""

    index: dict[str, ClassifiedMemoryStorePlugin]
    load_report: DomainPluginLoadReport

    @classmethod
    def from_discovery(cls, result: MemoryStorePluginDiscoveryResult) -> MemoryStorePluginCatalog:
        return cls(
            index=index_classified_memory_store_plugins(result.plugins),
            load_report=result.load_report,
        )


def discover_classified_memory_store_plugins(
    *,
    discover_entry_points: bool = True,
    explicit_plugins: Sequence[type] = (),
    on_load_failure: LoadIsolation = "isolate",
) -> MemoryStorePluginDiscoveryResult:
    """Discover and classify Memory store plugin candidates."""
    classified: list[ClassifiedMemoryStorePlugin] = []

    for plugin_type in explicit_plugins:
        record = classify_memory_store_plugin_record(plugin_type)
        if record is not None:
            classified.append(record)

    if not discover_entry_points:
        return MemoryStorePluginDiscoveryResult(
            plugins=tuple(classified),
            load_report=DomainPluginLoadReport.empty(EP_MEMORY_STORES),
        )

    accepted: list[EntryPointSpec] = []
    rejected: list[PluginAdmissionRejection] = []
    failed: list[EntryPointLoadResult] = []

    for result in load_entry_point_targets(
        EP_MEMORY_STORES,
        on_load_failure=on_load_failure,
    ):
        if result.error is not None:
            failed.append(result)
            continue
        try:
            plugin_type = resolve_entry_point_plugin_type(result.target, result.spec.value)
        except PluginLoadError as exc:
            if on_load_failure == "fail_fast":
                raise
            failed.append(EntryPointLoadResult(spec=result.spec, error=exc))
            continue
        kind = classify_memory_store_plugin(plugin_type)
        if kind is None:
            message = (
                f"Memory store entry point {result.spec.name!r} "
                "does not implement a supported Memory store plugin contract"
            )
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=message,
                    fail_closed=True,
                )
            )
            continue
        record = classify_memory_store_plugin_record(
            plugin_type,
            entry_point_name=result.spec.name,
            entry_point_spec=result.spec,
        )
        if record is None:
            message = (
                f"Memory store entry point {result.spec.name!r} "
                "returned an invalid or missing plugin_id"
            )
            rejected.append(
                PluginAdmissionRejection(
                    spec=result.spec,
                    reason_code=PluginAdmissionReasonCode.INVALID_TARGET_TYPE,
                    reason=message,
                    plugin_id=_safe_plugin_id(plugin_type),
                    fail_closed=True,
                )
            )
            continue
        classified.append(record)
        accepted.append(result.spec)

    load_report = DomainPluginLoadReport(
        group=EP_MEMORY_STORES,
        accepted=tuple(sorted(accepted, key=lambda spec: (spec.name, spec.value))),
        rejected=tuple(
            sorted(rejected, key=lambda item: (item.spec.name, item.spec.value))
        ),
        failed=tuple(sorted(failed, key=lambda item: (item.spec.name, item.spec.value))),
        registered_count=len(accepted),
    )
    return MemoryStorePluginDiscoveryResult(
        plugins=tuple(classified),
        load_report=load_report,
    )


def _safe_plugin_id(plugin_type: type) -> str | None:
    try:
        raw = plugin_type.plugin_id()
    except Exception:
        return None
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return None


def index_classified_memory_store_plugins(
    plugins: Sequence[ClassifiedMemoryStorePlugin],
) -> dict[str, ClassifiedMemoryStorePlugin]:
    """Index classified plugins by ``plugin_id``; duplicate ids fail closed."""
    index: dict[str, ClassifiedMemoryStorePlugin] = {}
    for record in plugins:
        existing = index.get(record.plugin_id)
        if existing is not None:
            raise MemoryStorePluginResolutionError(
                f"Duplicate memory store plugin_id {record.plugin_id!r} "
                f"({existing.entry_point_name!r} vs {record.entry_point_name!r})"
            )
        index[record.plugin_id] = record
    return index


def find_failed_entry_point_for_plugin_id(
    catalog: MemoryStorePluginCatalog,
    plugin_id: str,
) -> EntryPointLoadResult | None:
    """Return an isolated EP load failure matching ``plugin_id`` when present."""
    for item in catalog.load_report.failed:
        if item.spec.name == plugin_id:
            return item
    for record in catalog.index.values():
        if record.plugin_id == plugin_id and record.entry_point_name is not None:
            for item in catalog.load_report.failed:
                if item.spec.name == record.entry_point_name:
                    return item
    return None


def find_rejected_entry_point_for_plugin_id(
    catalog: MemoryStorePluginCatalog,
    plugin_id: str,
) -> PluginAdmissionRejection | None:
    """Return an admission rejection matching ``plugin_id`` when present."""
    for item in catalog.load_report.rejected:
        if item.spec.name == plugin_id or item.plugin_id == plugin_id:
            return item
    return None
