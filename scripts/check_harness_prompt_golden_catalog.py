#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Verify harness-critical prompt YAML content against golden hashes (FAUDIT-PE.1)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "agents", ROOT / "applications"):
    path_value = str(path)
    if path_value not in sys.path:
        sys.path.insert(0, path_value)

from intergrax.runtime.architecture.prompt_golden_catalog import (
    load_golden_expectations,
    verify_prompt_golden_catalog,
)

DEFAULT_CATALOG = ROOT / "prompts"
DEFAULT_EXPECTATIONS = ROOT / "tests" / "fixtures" / "prompt_golden" / "expectations.json"


def main() -> int:
    expectations = load_golden_expectations(DEFAULT_EXPECTATIONS)
    report = verify_prompt_golden_catalog(
        catalog_dir=DEFAULT_CATALOG,
        expectations=expectations,
    )
    if not report.passed:
        for failure in report.failures:
            print(failure, file=sys.stderr)
        return 1
    print(f"prompt golden catalog: ok ({report.checked} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
