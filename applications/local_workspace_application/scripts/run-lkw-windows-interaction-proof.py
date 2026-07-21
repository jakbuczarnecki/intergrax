#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Compatibility shim — delegates to the shared OS interaction proof runner."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SHARED = _SCRIPT_DIR / "run-lkw-os-interaction-proof.py"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--os-family" not in args:
        args = ["--os-family", "windows", *args]
    # Import shared module by path so this file stays a thin shim.
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "lkw_os_interaction_proof_shared", _SHARED
    )
    if spec is None or spec.loader is None:
        print("proof_result=FAIL")
        print("failure_reason=shared_proof_runner_missing")
        return 1
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(args))


if __name__ == "__main__":
    sys.exit(main())
