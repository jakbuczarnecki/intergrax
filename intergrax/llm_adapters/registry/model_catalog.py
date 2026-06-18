# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Central ModelCatalog — context windows and capability metadata (ADR-LLM-002)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

_BUNDLED_CATALOG_PATH = Path(__file__).with_name("model_catalog.yaml")
_CATALOG_ENV = "INTERGRAX_LLM_MODEL_CATALOG_PATH"


@dataclass(frozen=True, slots=True)
class ModelRecord:
    """Frozen catalog entry for a vendor model id."""

    model_id: str
    context_window_tokens: int
    supports_vision: bool = False
    supports_tools: bool = True
    supports_structured_output: bool = False
    provider_hints: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PrefixRule:
    """Prefix-based context window rule (longest match wins)."""

    prefix: str
    context_window_tokens: int


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """Immutable model metadata registry loaded from bundled + optional overlay YAML."""

    models: tuple[ModelRecord, ...]
    prefix_rules: tuple[PrefixRule, ...]
    provider_defaults: Mapping[str, int]
    fallback_default: int

    def lookup_exact(self, model_id: str) -> ModelRecord | None:
        normalized = (model_id or "").strip()
        if not normalized:
            return None
        for record in self.models:
            if record.model_id == normalized:
                return record
        return None

    def lookup_prefix(self, model_id: str) -> int | None:
        normalized = (model_id or "").strip()
        if not normalized:
            return None
        best: PrefixRule | None = None
        for rule in self.prefix_rules:
            if normalized.startswith(rule.prefix):
                if best is None or len(rule.prefix) > len(best.prefix):
                    best = rule
        return best.context_window_tokens if best is not None else None

    def provider_default(self, provider: str) -> int | None:
        key = (provider or "").strip().lower()
        if not key:
            return None
        value = self.provider_defaults.get(key)
        return int(value) if value is not None else None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ModelCatalog:
        models_raw = payload.get("models") or []
        prefix_raw = payload.get("prefix_rules") or []
        provider_defaults_raw = payload.get("provider_defaults") or {}
        fallback = int(payload.get("fallback_default", 32_000))

        models: list[ModelRecord] = []
        for row in models_raw:
            if not isinstance(row, dict):
                continue
            model_id = str(row.get("model_id", "")).strip()
            if not model_id:
                continue
            hints = row.get("provider_hints") or ()
            if isinstance(hints, str):
                hints = (hints,)
            models.append(
                ModelRecord(
                    model_id=model_id,
                    context_window_tokens=int(row["context_window_tokens"]),
                    supports_vision=bool(row.get("supports_vision", False)),
                    supports_tools=bool(row.get("supports_tools", True)),
                    supports_structured_output=bool(row.get("supports_structured_output", False)),
                    provider_hints=tuple(str(h) for h in hints),
                )
            )

        prefix_rules: list[PrefixRule] = []
        for row in prefix_raw:
            if not isinstance(row, dict):
                continue
            prefix = str(row.get("prefix", "")).strip()
            if not prefix:
                continue
            prefix_rules.append(
                PrefixRule(prefix=prefix, context_window_tokens=int(row["context_window_tokens"]))
            )
        prefix_rules.sort(key=lambda r: len(r.prefix), reverse=True)

        provider_defaults: dict[str, int] = {}
        if isinstance(provider_defaults_raw, Mapping):
            for key, value in provider_defaults_raw.items():
                provider_defaults[str(key).strip().lower()] = int(value)

        return cls(
            models=tuple(models),
            prefix_rules=tuple(prefix_rules),
            provider_defaults=provider_defaults,
            fallback_default=fallback,
        )

    @classmethod
    def load_yaml(cls, path: Path) -> ModelCatalog:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load ModelCatalog YAML")
        text = path.read_text(encoding="utf-8")
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"ModelCatalog YAML must be a mapping: {path}")
        return cls.from_mapping(payload)

    @classmethod
    def load_default(cls) -> ModelCatalog:
        base = cls.load_yaml(_BUNDLED_CATALOG_PATH)
        overlay_path = os.getenv(_CATALOG_ENV, "").strip()
        if not overlay_path:
            return base
        overlay = cls.load_yaml(Path(overlay_path))
        return cls._merge(base, overlay)

    @classmethod
    def _merge(cls, base: ModelCatalog, overlay: ModelCatalog) -> ModelCatalog:
        by_id = {record.model_id: record for record in base.models}
        for record in overlay.models:
            by_id[record.model_id] = record
        prefix_by_key = {rule.prefix: rule for rule in base.prefix_rules}
        for rule in overlay.prefix_rules:
            prefix_by_key[rule.prefix] = rule
        merged_prefix = tuple(
            sorted(prefix_by_key.values(), key=lambda r: len(r.prefix), reverse=True)
        )
        provider_defaults = dict(base.provider_defaults)
        provider_defaults.update(overlay.provider_defaults)
        fallback = overlay.fallback_default if overlay.fallback_default else base.fallback_default
        return cls(
            models=tuple(by_id.values()),
            prefix_rules=merged_prefix,
            provider_defaults=provider_defaults,
            fallback_default=fallback,
        )


def reset_model_catalog_cache() -> None:
    """Clear cached catalog (tests only)."""
    get_model_catalog.cache_clear()


@lru_cache(maxsize=1)
def get_model_catalog() -> ModelCatalog:
    return ModelCatalog.load_default()


def lookup_model_record(model_id: str, *, catalog: ModelCatalog | None = None) -> ModelRecord | None:
    cat = catalog or get_model_catalog()
    return cat.lookup_exact(model_id)
