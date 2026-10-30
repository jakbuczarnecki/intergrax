# © Artur Czarnecki. All rights reserved.

"""Execute registered current-HEAD qualification proofs (not hasattr / import-only checks)."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any


def load_proof_callable(module_path: str, function_name: str) -> Callable[..., Any]:
    module = importlib.import_module(module_path)
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise AssertionError(f"missing proof callable: {module_path}::{function_name}")
    return fn


def execute_proof(module_path: str, function_name: str, **kwargs: Any) -> None:
    fn = load_proof_callable(module_path, function_name)
    fn(**kwargs)
