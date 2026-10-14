# © Artur Czarnecki. All rights reserved.

"""Shared process-lifecycle store cleanup helpers (production + reference openers)."""

from __future__ import annotations

from collections.abc import Callable

from intergrax.applications._shared.production_execution_continuation_store import (
    SerializedSealExecutionContinuationStateStore,
)
from intergrax.runtime.enterprise_reliability.sqlite_provider_invocation_store import (
    SQLiteProviderInvocationStore,
)


def _close_owned_resource(
    close: Callable[[], None],
    *,
    mark_closed: Callable[[], None],
) -> BaseException | None:
    try:
        close()
    except BaseException as exc:
        return exc
    mark_closed()
    return None


def release_lifecycle_owned_stores(
    *,
    continuation_store: SerializedSealExecutionContinuationStateStore | None,
    owned_invocation: SQLiteProviderInvocationStore | None,
    continuation_closed: bool,
    invocation_closed: bool,
) -> list[BaseException]:
    errors: list[BaseException] = []
    if continuation_store is not None and not continuation_closed:
        err = _close_owned_resource(
            continuation_store.close,
            mark_closed=lambda: None,
        )
        if err is not None:
            errors.append(err)
    if owned_invocation is not None and not invocation_closed:
        err = _close_owned_resource(
            owned_invocation.close,
            mark_closed=lambda: None,
        )
        if err is not None:
            errors.append(err)
    return errors


def raise_grouped_exceptions(
    message: str,
    exceptions: list[BaseException],
    *,
    from_exc: BaseException | None = None,
) -> None:
    if all(isinstance(exc, Exception) for exc in exceptions):
        exception_only = [exc for exc in exceptions if isinstance(exc, Exception)]
        raise ExceptionGroup(message, exception_only) from from_exc
    raise BaseExceptionGroup(message, exceptions) from from_exc


def raise_primary_with_cleanup_errors(
    primary_exc: BaseException,
    cleanup_errors: list[BaseException],
    *,
    message: str,
) -> None:
    if cleanup_errors:
        raise_grouped_exceptions(
            message,
            [primary_exc, *cleanup_errors],
            from_exc=primary_exc,
        )
    raise primary_exc


def raise_startup_failure_with_cleanup(
    startup_exc: BaseException,
    cleanup_errors: list[BaseException],
    *,
    resource_scope: str,
) -> None:
    raise_primary_with_cleanup_errors(
        startup_exc,
        cleanup_errors,
        message=(
            f"{resource_scope} process lifecycle startup failed; "
            "errors while releasing acquired resources"
        ),
    )


def cleanup_errors_from_close_failure(cleanup_exc: BaseException) -> list[BaseException]:
    if isinstance(cleanup_exc, BaseExceptionGroup):
        return list(cleanup_exc.exceptions)
    return [cleanup_exc]


__all__ = [
    "cleanup_errors_from_close_failure",
    "raise_primary_with_cleanup_errors",
    "raise_startup_failure_with_cleanup",
    "release_lifecycle_owned_stores",
]
