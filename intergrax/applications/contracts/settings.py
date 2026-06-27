# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Application host settings contract for scaffolded Tier-3 applications."""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields
from typing import ClassVar

from intergrax.fastapi_core.config import ApiEnvironment


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


@dataclass(frozen=True, kw_only=True)
class IntergraxApplicationSettingsBase:
    """Platform-owned application host settings (Template Method for ``from_env``)."""

    env_prefix: ClassVar[str] = "APP_"

    environment: ApiEnvironment = ApiEnvironment.DEV
    route_prefix: str = "/v1/app"
    backend_host: str = "127.0.0.1"
    backend_port: int = 8091
    include_interaction_routes: bool = True
    interaction_route_prefix: str = "/v1/interactions"
    include_scheduler: bool = True
    scheduler_poll_seconds: float | None = None
    interaction_surface: str = "auto"
    include_mcp: bool = False
    mcp_mount_path: str = "/mcp"
    include_task_control: bool = True
    include_queue_worker: bool = True
    task_control_route_prefix: str = "/v1/tasks"

    @classmethod
    def from_env(cls) -> IntergraxApplicationSettingsBase:
        env = EnvReader(cls.env_prefix)
        values: dict[str, object] = {}
        values.update(cls._load_platform_env(env))
        values.update(cls._load_app_env(env))
        return cls(**values)

    @classmethod
    def _load_platform_env(cls, env: EnvReader) -> dict[str, object]:
        env_raw = (
            env.optional_str("BACKEND_ENV")
            or (os.environ.get("INTERGRAX_ENV") or "dev").strip().lower()
        )
        try:
            environment = ApiEnvironment(env_raw)
        except ValueError as exc:
            raise ValueError(
                f"{cls.env_prefix}BACKEND_ENV must be one of "
                f"{[item.value for item in ApiEnvironment]}, got {env_raw!r}."
            ) from exc

        poll_raw = (os.environ.get("INTERGRAX_SCHEDULER_POLL_SECONDS") or "").strip()
        scheduler_poll = float(poll_raw) if poll_raw else None

        return {
            "environment": environment,
            "route_prefix": env.str("ROUTE_PREFIX", default=cls._field_default("route_prefix")),  # type: ignore[arg-type]
            "backend_host": env.str("BACKEND_HOST", default=cls._field_default("backend_host")),  # type: ignore[arg-type]
            "backend_port": env.int("BACKEND_PORT", default=cls._field_default("backend_port")),  # type: ignore[arg-type]
            "include_interaction_routes": env.bool(
                "INCLUDE_INTERACTIONS",
                default=cls._field_default("include_interaction_routes"),  # type: ignore[arg-type]
            ),
            "interaction_route_prefix": env.str(
                "INTERACTION_ROUTE_PREFIX",
                default=cls._field_default("interaction_route_prefix"),  # type: ignore[arg-type]
            ),
            "include_scheduler": env.bool(
                "INCLUDE_SCHEDULER",
                default=cls._field_default("include_scheduler"),  # type: ignore[arg-type]
            ),
            "scheduler_poll_seconds": scheduler_poll,
            "interaction_surface": env.str(
                "INTERACTION_SURFACE",
                default=cls._field_default("interaction_surface"),  # type: ignore[arg-type]
            ).lower()
            or "auto",
            "include_mcp": env.bool("INCLUDE_MCP", default=cls._field_default("include_mcp")),  # type: ignore[arg-type]
            "mcp_mount_path": env.str("MCP_MOUNT_PATH", default=cls._field_default("mcp_mount_path")),  # type: ignore[arg-type]
            "include_task_control": env.bool(
                "INCLUDE_TASK_CONTROL",
                default=cls._field_default("include_task_control"),  # type: ignore[arg-type]
            ),
            "include_queue_worker": env.bool(
                "INCLUDE_QUEUE_WORKER",
                default=cls._field_default("include_queue_worker"),  # type: ignore[arg-type]
            ),
            "task_control_route_prefix": env.str(
                "TASK_CONTROL_ROUTE_PREFIX",
                default=cls._field_default("task_control_route_prefix"),  # type: ignore[arg-type]
            ),
        }

    @classmethod
    def _load_app_env(cls, env: EnvReader) -> dict[str, object]:
        _ = env
        return {}

    @classmethod
    def _field_default(cls, name: str) -> object:
        for field in fields(cls):
            if field.name != name:
                continue
            if field.default is not MISSING:
                return field.default
            if field.default_factory is not MISSING:  # type: ignore[attr-defined]
                return field.default_factory()
        raise KeyError(name)
