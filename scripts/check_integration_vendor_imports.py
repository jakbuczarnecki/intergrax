#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Entrypoint shim for integration vendor-import boundary audit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def main() -> int:
    script = Path(__file__).resolve().parent / "maintenance" / "check_integration_vendor_imports.py"
    spec = importlib.util.spec_from_file_location("_check_integration_vendor_imports", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load vendor import checker from {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main())


if __name__ == "__main__":
    raise SystemExit(main())
