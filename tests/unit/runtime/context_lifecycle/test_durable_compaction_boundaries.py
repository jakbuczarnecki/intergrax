# © Artur Czarnecki. All rights reserved.

"""TOKEN-10E-1 durable compaction non-implementation boundary guards."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_RUNTIME_SYMBOLS = (
    "DurableCompactionRepository",
    "DurableCompactionCoordinator",
    "DurableCompactionWorker",
    "activate_durable_compaction",
    "compact_ledger_in_place",
)


def test_context_lifecycle_package_does_not_export_forbidden_runtime_symbols() -> None:
    import intergrax.runtime.context_lifecycle as context_lifecycle

    for symbol in _FORBIDDEN_RUNTIME_SYMBOLS:
        assert symbol not in context_lifecycle.__all__
        assert not hasattr(context_lifecycle, symbol)


def test_context_lifecycle_modules_do_not_define_forbidden_runtime_symbols() -> None:
    package = importlib.import_module("intergrax.runtime.context_lifecycle")
    for module_info in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        module = importlib.import_module(module_info.name)
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        for symbol in _FORBIDDEN_RUNTIME_SYMBOLS:
            assert not hasattr(module, symbol), (
                f"{module_info.name} must not define forbidden runtime symbol {symbol!r}"
            )
