# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Hosting policy contracts and standard presets (APP-HOST-1E)."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from intergrax.hosting.contracts.public_data import (
    derive_stable_callable_id,
    validate_bounded_identifier,
    validate_non_negative_bounded_seconds,
    validate_positive_bounded_seconds,
)


class ShutdownStrategy(str, Enum):
    DRAIN_THEN_CANCEL = "drain_then_cancel"
    CANCEL_IMMEDIATELY = "cancel_immediately"
    WAIT_UNTIL_COMPLETE = "wait_until_complete"


class RestartMode(str, Enum):
    NEVER = "never"
    ON_FAILURE = "on_failure"
    ALWAYS = "always"
    CUSTOM = "custom"


class ComponentFailureAction(str, Enum):
    FAIL_HOST = "fail_host"
    MARK_NOT_READY = "mark_not_ready"
    MARK_DEGRADED = "mark_degraded"
    RESTART_COMPONENT = "restart_component"
    REQUEST_PROCESS_RESTART = "request_process_restart"
    IGNORE_WITH_DIAGNOSTIC = "ignore_with_diagnostic"


class BlockingHookFailureAction(str, Enum):
    FAIL_STARTUP = "fail_startup"
    FAIL_READINESS = "fail_readiness"
    CONTINUE_SHUTDOWN = "continue_shutdown"


class ObserverHookFailureAction(str, Enum):
    DIAGNOSTIC_ONLY = "diagnostic_only"
    PRESERVE_ORIGINAL_FAILURE = "preserve_original_failure"


class InstanceExclusivityMode(str, Enum):
    SINGLE_INSTANCE = "single_instance"
    MULTI_INSTANCE = "multi_instance"


class LifecyclePolicy(BaseModel):
    """Lifecycle timing and readiness stabilization settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    default_blocking_hook_timeout_seconds: float = 30.0
    default_observer_hook_timeout_seconds: float = 5.0
    component_startup_concurrency: int = 4
    startup_compensation_enabled: bool = True
    readiness_stabilization_seconds: float = 0.0

    @field_validator("default_blocking_hook_timeout_seconds", "default_observer_hook_timeout_seconds")
    @classmethod
    def _validate_hook_timeouts(cls, value: float) -> float:
        return validate_positive_bounded_seconds(value, field_name="hook_timeout_seconds")

    @field_validator("readiness_stabilization_seconds")
    @classmethod
    def _validate_stabilization(cls, value: float) -> float:
        if value < 0:
            raise ValueError("readiness_stabilization_seconds must be non-negative")
        if value > 300.0:
            raise ValueError("readiness_stabilization_seconds must be at most 300 seconds")
        return value

    @field_validator("component_startup_concurrency")
    @classmethod
    def _validate_concurrency(cls, value: int) -> int:
        if value < 1 or value > 64:
            raise ValueError("component_startup_concurrency must be between 1 and 64")
        return value

    @classmethod
    def standard(cls) -> LifecyclePolicy:
        return cls()


class ShutdownPolicy(BaseModel):
    """Bounded shutdown strategy and timeout configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: ShutdownStrategy
    drain_timeout_seconds: float
    cancel_timeout_seconds: float
    flush_timeout_seconds: float

    @field_validator("drain_timeout_seconds", "cancel_timeout_seconds")
    @classmethod
    def _validate_optional_shutdown_timeouts(cls, value: float) -> float:
        return validate_non_negative_bounded_seconds(value, field_name="shutdown_timeout_seconds")

    @field_validator("flush_timeout_seconds")
    @classmethod
    def _validate_flush_timeout(cls, value: float) -> float:
        return validate_positive_bounded_seconds(value, field_name="flush_timeout_seconds")

    @model_validator(mode="after")
    def _validate_strategy_timeouts(self) -> ShutdownPolicy:
        if self.strategy is ShutdownStrategy.CANCEL_IMMEDIATELY and self.drain_timeout_seconds != 0:
            raise ValueError("cancel_immediately requires drain_timeout_seconds=0")
        if self.strategy is ShutdownStrategy.WAIT_UNTIL_COMPLETE and self.cancel_timeout_seconds != 0:
            raise ValueError("wait_until_complete requires cancel_timeout_seconds=0")
        return self

    @classmethod
    def standard(cls) -> ShutdownPolicy:
        return cls.drain_then_cancel()

    @classmethod
    def drain_then_cancel(
        cls,
        *,
        drain_timeout_seconds: float = 30.0,
        cancel_timeout_seconds: float = 10.0,
        flush_timeout_seconds: float = 5.0,
    ) -> ShutdownPolicy:
        return cls(
            strategy=ShutdownStrategy.DRAIN_THEN_CANCEL,
            drain_timeout_seconds=drain_timeout_seconds,
            cancel_timeout_seconds=cancel_timeout_seconds,
            flush_timeout_seconds=flush_timeout_seconds,
        )

    @classmethod
    def cancel_immediately(
        cls,
        *,
        cancel_timeout_seconds: float = 10.0,
        flush_timeout_seconds: float = 5.0,
    ) -> ShutdownPolicy:
        return cls(
            strategy=ShutdownStrategy.CANCEL_IMMEDIATELY,
            drain_timeout_seconds=0.0,
            cancel_timeout_seconds=cancel_timeout_seconds,
            flush_timeout_seconds=flush_timeout_seconds,
        )

    @classmethod
    def wait_until_complete(
        cls,
        *,
        drain_timeout_seconds: float = 300.0,
        flush_timeout_seconds: float = 5.0,
    ) -> ShutdownPolicy:
        return cls(
            strategy=ShutdownStrategy.WAIT_UNTIL_COMPLETE,
            drain_timeout_seconds=drain_timeout_seconds,
            cancel_timeout_seconds=0.0,
            flush_timeout_seconds=flush_timeout_seconds,
        )


class RestartPolicy(BaseModel):
    """Restart classification and bounded backoff configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    mode: RestartMode
    max_attempts: int = 3
    attempt_window_seconds: float = 300.0
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.1
    reset_after_stable_seconds: float = 300.0
    custom_classifier: SkipJsonSchema[Callable[..., bool] | None] = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    custom_classifier_id: str | None = None

    @field_validator("max_attempts")
    @classmethod
    def _validate_max_attempts(cls, value: int) -> int:
        if value < 0 or value > 100:
            raise ValueError("max_attempts must be between 0 and 100")
        return value

    @field_validator(
        "attempt_window_seconds",
        "initial_backoff_seconds",
        "max_backoff_seconds",
        "reset_after_stable_seconds",
    )
    @classmethod
    def _validate_positive_seconds(cls, value: float) -> float:
        return validate_positive_bounded_seconds(value, field_name="restart_seconds")

    @field_validator("multiplier")
    @classmethod
    def _validate_multiplier(cls, value: float) -> float:
        if value < 1.0 or value > 10.0:
            raise ValueError("multiplier must be between 1.0 and 10.0")
        return value

    @field_validator("jitter_ratio")
    @classmethod
    def _validate_jitter(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("jitter_ratio must be between 0.0 and 1.0")
        return value

    @field_validator("custom_classifier_id")
    @classmethod
    def _validate_custom_classifier_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return validate_bounded_identifier(value, field_name="custom_classifier_id")

    @model_validator(mode="after")
    def _resolve_custom_classifier(self) -> RestartPolicy:
        if self.mode is RestartMode.CUSTOM:
            if self.custom_classifier_id is None and self.custom_classifier is None:
                raise ValueError("custom restart mode requires custom_classifier or custom_classifier_id")
            if self.custom_classifier_id is None and self.custom_classifier is not None:
                derived_id = derive_stable_callable_id(
                    self.custom_classifier,
                    field_name="custom_classifier_id",
                )
                object.__setattr__(self, "custom_classifier_id", derived_id)
        elif self.custom_classifier is not None:
            raise ValueError("custom_classifier is only valid for custom restart mode")
        return self

    def to_public_policy(self) -> RestartPolicy:
        """Return a public-only restart policy without runtime callables."""
        data = self.public_dict()
        return RestartPolicy.model_validate(data)

    @classmethod
    def never(cls) -> RestartPolicy:
        return cls(mode=RestartMode.NEVER, max_attempts=0)

    @classmethod
    def on_failure(cls, *, max_attempts: int = 3) -> RestartPolicy:
        return cls(mode=RestartMode.ON_FAILURE, max_attempts=max_attempts)

    @classmethod
    def always(cls, *, max_attempts: int = 100) -> RestartPolicy:
        return cls(mode=RestartMode.ALWAYS, max_attempts=max_attempts)

    def public_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "mode": self.mode.value,
            "max_attempts": self.max_attempts,
            "attempt_window_seconds": self.attempt_window_seconds,
            "initial_backoff_seconds": self.initial_backoff_seconds,
            "max_backoff_seconds": self.max_backoff_seconds,
            "multiplier": self.multiplier,
            "jitter_ratio": self.jitter_ratio,
            "reset_after_stable_seconds": self.reset_after_stable_seconds,
        }
        if self.custom_classifier_id is not None:
            payload["custom_classifier_id"] = self.custom_classifier_id
        return payload


class ComponentFailurePolicy(BaseModel):
    """Default component failure actions for required and optional components."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    required_component_action: ComponentFailureAction = ComponentFailureAction.FAIL_HOST
    optional_component_action: ComponentFailureAction = ComponentFailureAction.MARK_DEGRADED

    @classmethod
    def standard(cls) -> ComponentFailurePolicy:
        return cls()


class HookFailurePolicy(BaseModel):
    """Fixed hosting hook failure semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    before_start: Literal[BlockingHookFailureAction.FAIL_STARTUP] = (
        BlockingHookFailureAction.FAIL_STARTUP
    )
    before_ready: Literal[BlockingHookFailureAction.FAIL_READINESS] = (
        BlockingHookFailureAction.FAIL_READINESS
    )
    before_stop: Literal[BlockingHookFailureAction.CONTINUE_SHUTDOWN] = (
        BlockingHookFailureAction.CONTINUE_SHUTDOWN
    )
    observer: Literal[ObserverHookFailureAction.DIAGNOSTIC_ONLY] = (
        ObserverHookFailureAction.DIAGNOSTIC_ONLY
    )
    on_failure: Literal[ObserverHookFailureAction.PRESERVE_ORIGINAL_FAILURE] = (
        ObserverHookFailureAction.PRESERVE_ORIGINAL_FAILURE
    )

    @classmethod
    def standard(cls) -> HookFailurePolicy:
        return cls()


class InstancePolicy(BaseModel):
    """Platform-neutral instance scope and exclusivity policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exclusivity_mode: InstanceExclusivityMode = InstanceExclusivityMode.SINGLE_INSTANCE
    allow_stale_recovery: bool = True

    @classmethod
    def standard(cls) -> InstancePolicy:
        return cls()
