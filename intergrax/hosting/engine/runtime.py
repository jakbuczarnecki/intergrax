# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Opaque hosted application runtime factory invocation (APP-HOST-2F)."""

from __future__ import annotations

import inspect
from collections.abc import Callable

from intergrax.hosting.contracts.context import HostedApplicationContext
from intergrax.hosting.engine.ports import HostedApplicationRuntime
from intergrax.hosting.errors import HostedApplicationConfigurationError, HostedApplicationRuntimeError


_SUPPORTED_PARAM_COUNTS = frozenset({0, 1})


def _validate_factory_signature(factory: Callable[..., object]) -> inspect.Signature:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError) as exc:
        raise HostedApplicationConfigurationError(
            "application factory signature cannot be inspected"
        ) from exc
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]
    if len(parameters) not in _SUPPORTED_PARAM_COUNTS:
        raise HostedApplicationConfigurationError(
            "application factory must accept zero arguments or a single context argument"
        )
    if len(parameters) == 1 and parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise HostedApplicationConfigurationError(
            "application factory context argument must be positional"
        )
    return signature


async def invoke_application_factory(
    factory: Callable[..., object],
    context: HostedApplicationContext,
) -> HostedApplicationRuntime:
    """Invoke a profile application factory and validate the opaque runtime result."""
    signature = _validate_factory_signature(factory)
    parameters = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]
    try:
        if len(parameters) == 0:
            result = factory()
        else:
            result = factory(context)
    except Exception as exc:
        raise HostedApplicationRuntimeError("application factory invocation failed") from exc

    if inspect.isawaitable(result):
        try:
            result = await result
        except Exception as exc:
            raise HostedApplicationRuntimeError("application factory await failed") from exc

    if not isinstance(result, HostedApplicationRuntime):
        raise HostedApplicationConfigurationError(
            "application factory must return a HostedApplicationRuntime implementation"
        )
    return result
