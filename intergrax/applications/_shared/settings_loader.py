# © Artur Czarnecki. All rights reserved.

"""Host-owned environment loading for application settings DTOs."""

from __future__ import annotations

import os
from dataclasses import MISSING, fields
from typing import ClassVar, TypeVar

from intergrax.applications.contracts.settings import IntergraxApplicationSettingsBase
from intergrax.contracts.api_environment import ApiEnvironment
from intergrax.knowledge.contracts.validation import JsonValue

SettingsT = TypeVar("SettingsT", bound=IntergraxApplicationSettingsBase)


class EnvReader:
    """Prefix-aware environment variable reader for application settings."""

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def raw(self, name: str, default: str | None = None) -> str | None:
        value = os.environ.get(f"{self._prefix}{name}")
        if value is None:
            return default
        return value

    def optional_str(self, name: str) -> str | None:
        raw = self.raw(name)
        if raw is None:
            return None
        stripped = raw.strip()
        return stripped or None

    def str(self, name: str, *, default: str) -> str:
        raw = self.optional_str(name)
        if raw is None:
            return default
        return raw

    def int(self, name: str, *, default: int) -> int:
        raw = self.optional_str(name)
        if raw is None:
            return default
        return int(raw)

    def float(self, name: str, *, default: float) -> float:
        raw = self.optional_str(name)
        if raw is None:
            return default
        return float(raw)

    def bool(self, name: str, *, default: bool = False) -> bool:
        raw = self.raw(name)
        if raw is None:
            return default
        return raw.strip().lower() not in {"0", "false", "no", "off"}

    def csv_set(self, name: str) -> frozenset[str]:
        raw = self.optional_str(name)
        if raw is None:
            return frozenset()
        return frozenset(part.strip() for part in raw.split(",") if part.strip())


class ApplicationSettingsEnvHost:
    """Mixin for Tier-3 host settings: ``from_env`` and app-specific env hooks."""

    env_prefix: ClassVar[str] = "APP_"

    @classmethod
    def from_env(cls: type[SettingsT]) -> SettingsT:
        return load_application_settings_from_env(cls)

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, JsonValue]:
        return {}


def _field_default(cls: type[IntergraxApplicationSettingsBase], name: str) -> JsonValue:
    for field in fields(cls):
        if field.name != name:
            continue
        if field.default is not MISSING:
            return field.default  # type: ignore[return-value]
        if field.default_factory is not MISSING:  # type: ignore[attr-defined]
            return field.default_factory()  # type: ignore[misc]
    raise KeyError(name)


def _load_platform_env(
    cls: type[IntergraxApplicationSettingsBase],
    env: EnvReader,
) -> dict[str, JsonValue]:
    env_raw = (
        env.optional_str("BACKEND_ENV")
        or (os.environ.get("INTERGRAX_ENV") or "dev").strip().lower()
    )
    if env_raw == "staging":
        env_raw = "stage"
    try:
        environment = ApiEnvironment(env_raw)
    except ValueError as exc:
        raise ValueError(
            f"{getattr(cls, 'env_prefix', 'APP_')}BACKEND_ENV must be one of "
            f"{[item.value for item in ApiEnvironment]}, got {env_raw!r}."
        ) from exc

    poll_raw = (os.environ.get("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
    scheduler_poll = float(poll_raw) if poll_raw else None

    return {
        "environment": environment,
        "route_prefix": env.str("ROUTE_PREFIX", default=str(_field_default(cls, "route_prefix"))),
        "backend_host": env.str("BACKEND_HOST", default=str(_field_default(cls, "backend_host"))),
        "backend_port": env.int(
            "BACKEND_PORT",
            default=int(_field_default(cls, "backend_port")),  # type: ignore[arg-type]
        ),
        "include_interaction_routes": env.bool(
            "INCLUDE_INTERACTIONS",
            default=bool(_field_default(cls, "include_interaction_routes")),
        ),
        "interaction_route_prefix": env.str(
            "INTERACTION_ROUTE_PREFIX",
            default=str(_field_default(cls, "interaction_route_prefix")),
        ),
        "include_scheduler": env.bool(
            "INCLUDE_SCHEDULER",
            default=bool(_field_default(cls, "include_scheduler")),
        ),
        "scheduler_poll_seconds": scheduler_poll,
        "interaction_surface": env.str(
            "INTERACTION_SURFACE",
            default=str(_field_default(cls, "interaction_surface")),
        ).lower()
        or "auto",
        "include_mcp": env.bool("INCLUDE_MCP", default=bool(_field_default(cls, "include_mcp"))),
        "mcp_mount_path": env.str(
            "MCP_MOUNT_PATH",
            default=str(_field_default(cls, "mcp_mount_path")),
        ),
        "include_task_control": env.bool(
            "INCLUDE_TASK_CONTROL",
            default=bool(_field_default(cls, "include_task_control")),
        ),
        "include_queue_worker": env.bool(
            "INCLUDE_QUEUE_WORKER",
            default=bool(_field_default(cls, "include_queue_worker")),
        ),
        "task_control_route_prefix": env.str(
            "TASK_CONTROL_ROUTE_PREFIX",
            default=str(_field_default(cls, "task_control_route_prefix")),
        ),
    }


def load_application_settings_from_env(cls: type[SettingsT]) -> SettingsT:
    env = EnvReader(getattr(cls, "env_prefix", "APP_"))
    values: dict[str, JsonValue] = {}
    values.update(_load_platform_env(cls, env))
    load_app_env = getattr(cls, "_load_app_env", None)
    if load_app_env is not None:
        values.update(load_app_env(env))
    return cls(**values)  # type: ignore[arg-type]


__all__ = [
    "ApplicationSettingsEnvHost",
    "EnvReader",
    "load_application_settings_from_env",
]
