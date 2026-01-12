# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.



# STARTUP EXAMPLE:
#
# import logging
# from intergrax.logging import IntergraxLogging
#
# IntergraxLogging.startup(
#     enabled=True,
#     level=logging.WARNING,
#     enable_console=True,
#     enable_runtime_events=False,
# )


# USAGE EXAMPLE:
#
# from intergrax.logging import IntergraxLogging
#
# logger = IntergraxLogging.get_logger(__name__, component="rag")
# logger.warning("[intergraxDocumentsLoader] Skipping large file (%.1f MB): %s", size_mb, file_path)
# logger.info("test")


from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Protocol, cast

from intergrax.utils.time_provider import SystemTimeProvider


__all__ = ["IntergraxLogging"]


# ============================
# Internal contracts / models
# ============================

class RuntimeStateLike(Protocol):
    """
    Minimal contract required for emitting events into runtime state.
    Your RuntimeState should already be compatible with this method signature.
    """
    def trace_event(
        self,
        *,
        component: str,
        step: str,
        message: str,
        data: Dict[str, Any],
        level: str,
    ) -> None: ...


@dataclass(frozen=True)
class LoggingConfig:
    enabled: bool = True
    level: int = logging.INFO
    enable_console: bool = True


@dataclass(frozen=True)
class LogContext:
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    step: Optional[str] = None
    component: Optional[str] = None


# ============================
# Internal state (contextvars)
# ============================

_global_cfg_var: ContextVar[LoggingConfig] = ContextVar(
    "intergrax_logging_global_cfg",
    default=LoggingConfig(),
)

_configured_var: ContextVar[bool] = ContextVar(
    "intergrax_logging_configured",
    default=False,
)

_log_ctx_var: ContextVar[LogContext] = ContextVar(
    "intergrax_log_ctx",
    default=LogContext(),
)

_runtime_state_var: ContextVar[Optional[RuntimeStateLike]] = ContextVar(
    "intergrax_runtime_state",
    default=None,
)


# ============================
# Internal record keys
# ============================

_KEY_RUN_ID = "ig_run_id"
_KEY_SESSION_ID = "ig_session_id"
_KEY_USER_ID = "ig_user_id"
_KEY_TENANT_ID = "ig_tenant_id"
_KEY_STEP = "ig_step"
_KEY_COMPONENT = "ig_component"
_KEY_DATA = "data"  # structured payload dict


# ============================
# Internal helpers
# ============================

def _is_enabled() -> bool:
    return _global_cfg_var.get().enabled


def _get_root_logger() -> logging.Logger:
    # Only Intergrax logger tree (never root)
    return logging.getLogger("intergrax")


# ============================
# Internal logging plumbing
# ============================

class _IntergraxContextFilter(logging.Filter):
    """
    Attaches Intergrax context fields to every LogRecord in the 'intergrax' logger tree.
    Uses record.__dict__ only (no getattr).
    Also respects the global enabled switch.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if not _is_enabled():
            return False

        ctx = _log_ctx_var.get()
        d = record.__dict__

        d.setdefault(_KEY_RUN_ID, ctx.run_id or "-")
        d.setdefault(_KEY_SESSION_ID, ctx.session_id or "-")
        d.setdefault(_KEY_USER_ID, ctx.user_id or "-")
        d.setdefault(_KEY_TENANT_ID, ctx.tenant_id or "-")
        d.setdefault(_KEY_STEP, ctx.step or "-")

        if _KEY_COMPONENT not in d:
            d[_KEY_COMPONENT] = ctx.component or "-"

        raw_data = d.get(_KEY_DATA)
        if raw_data is None:
            return True
        if isinstance(raw_data, dict):
            return True

        # Normalize non-dict payloads into dict.
        d[_KEY_DATA] = {"value": raw_data}
        return True


class _ConsoleFormatter(logging.Formatter):
    """
    Human-friendly, stable diagnostic format.
    """
    def format(self, record: logging.LogRecord) -> str:
        d = record.__dict__

        ts = SystemTimeProvider.utc_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"
        level = record.levelname

        component = cast(str, d.get(_KEY_COMPONENT, "-"))
        step = cast(str, d.get(_KEY_STEP, "-"))
        run_id = cast(str, d.get(_KEY_RUN_ID, "-"))

        msg = record.getMessage()
        line = f"{ts} [{level}] {component}:{step} (run={run_id}) {msg}"

        data = d.get(_KEY_DATA)
        if isinstance(data, dict) and data:
            try:
                line += " | data=" + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            except Exception:
                line += " | data=<unserializable>"

        return line


class _ComponentLoggerAdapter(logging.LoggerAdapter):
    """
    Adapter that enforces ig_component for all records produced by this logger instance.
    """
    def __init__(self, logger: logging.Logger, *, component: str) -> None:
        super().__init__(logger, extra={_KEY_COMPONENT: component})

    def process(self, msg: Any, kwargs: Mapping[str, Any]) -> tuple[Any, Dict[str, Any]]:
        out_kwargs: Dict[str, Any] = dict(kwargs)

        extra = out_kwargs.get("extra")
        if extra is None:
            extra_dict: Dict[str, Any] = {}
        elif isinstance(extra, dict):
            extra_dict = cast(Dict[str, Any], extra)
        else:
            extra_dict = {"value": extra}

        extra_dict[_KEY_COMPONENT] = cast(str, self.extra[_KEY_COMPONENT])
        out_kwargs["extra"] = extra_dict

        return msg, out_kwargs


def _configure_logging(*, force_reconfigure: bool = False) -> None:
    """
    Internal config for Intergrax logger tree.
    - Does NOT configure root logger.
    - Configures only 'intergrax' logger tree.
    """
    cfg = _global_cfg_var.get()

    already = _configured_var.get()
    if already and not force_reconfigure:
        return

    root = _get_root_logger()
    root.setLevel(cfg.level)
    root.propagate = False

    if force_reconfigure:
        for h in list(root.handlers):
            root.removeHandler(h)
        for f in list(root.filters):
            root.removeFilter(f)

    root.addFilter(_IntergraxContextFilter())

    if cfg.enable_console:
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(cfg.level)
        sh.setFormatter(_ConsoleFormatter())
        root.addHandler(sh)

    _configured_var.set(True)


def _build_logger_name(name: Optional[str]) -> str:
    base = "intergrax"

    if name is None or name == "":
        return base

    if name.startswith("intergrax."):
        return name

    return f"{base}.{name}"


def _get_logger(name: Optional[str] = None, *, component: Optional[str] = None) -> logging.Logger:
    """
    Internal logger factory.
    NOTE: Does NOT auto-configure; configuration is done by IntergraxLogging.startup(...)
    """
    logger_name = _build_logger_name(name)
    lg = logging.getLogger(logger_name)

    if component:
        return _ComponentLoggerAdapter(lg, component=component)

    return lg


# ============================
# Public facade (only export)
# ============================

class IntergraxLogging:
    """
    Intergrax logging facade (single public entrypoint).

    Typical application usage (entrypoint):
        import logging
        from intergrax.logging import IntergraxLogging

        IntergraxLogging.startup(
            enabled=True,
            level=logging.WARNING,
            enable_console=True,
            enable_runtime_events=False,
        )

    Typical module usage:
        from intergrax.logging import IntergraxLogging

        logger = IntergraxLogging.get_logger(__name__, component="rag")
        logger.warning("...")
    """

    # ---- Startup / policy ----

    @staticmethod
    def startup(
        *,
        enabled: bool = True,
        level: int = logging.INFO,
        enable_console: bool = True,
        force_reconfigure: bool = False,
    ) -> None:
        """
        Configure Intergrax logging once, from application entrypoint.

        - enabled=False cuts off ALL intergrax.* logs (filter returns False).
        - level controls which severities are emitted (DEBUG/INFO/WARNING/ERROR).
        - enable_console routes logs to stdout.
        - enable_runtime_events routes logs to RuntimeState.trace_event (when state is set).
        - force_reconfigure should be used in tests/notebooks when rerunning setup.
        """
        _global_cfg_var.set(
            LoggingConfig(
                enabled=enabled,
                level=level,
                enable_console=enable_console,
            )
        )
        _configure_logging(force_reconfigure=force_reconfigure)

    @staticmethod
    def reconfigure(*, force_reconfigure: bool = True) -> None:
        """
        Re-apply current configuration to handlers/filters.
        Use when you changed config after startup.
        """
        _configure_logging(force_reconfigure=force_reconfigure)

    @staticmethod
    def enable() -> None:
        cfg = _global_cfg_var.get()
        _global_cfg_var.set(
            LoggingConfig(
                enabled=True,
                level=cfg.level,
                enable_console=cfg.enable_console,
                enable_runtime_events=cfg.enable_runtime_events,
            )
        )

    @staticmethod
    def disable() -> None:
        cfg = _global_cfg_var.get()
        _global_cfg_var.set(
            LoggingConfig(
                enabled=False,
                level=cfg.level,
                enable_console=cfg.enable_console,
                enable_runtime_events=cfg.enable_runtime_events,
            )
        )

    @staticmethod
    def set_level(level: int) -> None:
        """
        Update severity threshold. If logging was already started, call reconfigure()
        (or call startup(..., force_reconfigure=True)) to update handler levels.
        """
        cfg = _global_cfg_var.get()
        _global_cfg_var.set(
            LoggingConfig(
                enabled=cfg.enabled,
                level=level,
                enable_console=cfg.enable_console,
                enable_runtime_events=cfg.enable_runtime_events,
            )
        )

    @staticmethod
    def set_config(
        *,
        enabled: Optional[bool] = None,
        level: Optional[int] = None,
        enable_console: Optional[bool] = None,
        enable_runtime_events: Optional[bool] = None,
    ) -> None:
        """
        Update configuration in memory. If you change handler-related settings
        (level/enable_console/enable_runtime_events) after startup, call reconfigure().
        """
        cfg = _global_cfg_var.get()
        _global_cfg_var.set(
            LoggingConfig(
                enabled=cfg.enabled if enabled is None else enabled,
                level=cfg.level if level is None else level,
                enable_console=cfg.enable_console if enable_console is None else enable_console,
                enable_runtime_events=cfg.enable_runtime_events if enable_runtime_events is None else enable_runtime_events,
            )
        )

    @staticmethod
    def get_config() -> LoggingConfig:
        return _global_cfg_var.get()

    # ---- Logger factory ----

    @staticmethod
    def get_logger(name: Optional[str] = None, *, component: Optional[str] = None) -> logging.Logger:
        return _get_logger(name, component=component)

    # ---- Context (correlation) ----

    @staticmethod
    def set_context(
        *,
        run_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        step: Optional[str] = None,
        component: Optional[str] = None,
    ) -> None:
        cur = _log_ctx_var.get()
        _log_ctx_var.set(
            LogContext(
                run_id=run_id if run_id is not None else cur.run_id,
                session_id=session_id if session_id is not None else cur.session_id,
                user_id=user_id if user_id is not None else cur.user_id,
                tenant_id=tenant_id if tenant_id is not None else cur.tenant_id,
                step=step if step is not None else cur.step,
                component=component if component is not None else cur.component,
            )
        )

    @staticmethod
    def clear_context() -> None:
        _log_ctx_var.set(LogContext())

    # ---- Runtime state sink ----

    @staticmethod
    def set_state(state: RuntimeStateLike) -> None:
        _runtime_state_var.set(state)

    @staticmethod
    def clear_state() -> None:
        _runtime_state_var.set(None)
