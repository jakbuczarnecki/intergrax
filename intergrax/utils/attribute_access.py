# © Artur Czarnecki. All rights reserved.

"""Centralized dynamic attribute access for vendor/SDK boundaries (no getattr at call sites)."""

from __future__ import annotations

from typing import TypeVar, overload

T = TypeVar("T")
_MISSING = object()


def _resolve_attr(obj: object, name: str) -> object:
    try:
        return object.__getattribute__(obj, name)
    except AttributeError as missing:
        try:
            dunder_getattr = object.__getattribute__(obj, "__getattr__")
        except AttributeError:
            raise missing
        try:
            return dunder_getattr(name)
        except AttributeError as exc:
            raise AttributeError(
                f"{type(obj).__name__!r} object has no attribute {name!r}"
            ) from exc


@overload
def optional(obj: object, name: str) -> object: ...


@overload
def optional(obj: object, name: str, default: T) -> object | T: ...


def optional(obj: object, name: str, default: object = _MISSING) -> object:
    try:
        return _resolve_attr(obj, name)
    except AttributeError:
        if default is _MISSING:
            raise
        return default


def optional_bool(obj: object, name: str, *, default: bool = False) -> bool:
    try:
        return bool(_resolve_attr(obj, name))
    except AttributeError:
        return default


def optional_str(obj: object, name: str, *, default: str = "") -> str:
    try:
        value = _resolve_attr(obj, name)
    except AttributeError:
        return default
    if value is None:
        return default
    return str(value)


def is_callable_attr(obj: object, name: str) -> bool:
    try:
        value = _resolve_attr(obj, name)
    except AttributeError:
        return False
    return callable(value)
