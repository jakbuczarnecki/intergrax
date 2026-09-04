#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""M-LLM-X.11.8 — Tier-3 resolver call sites must use routing context bridge."""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

WIRING_FILES = (
    REPO_ROOT / "intergrax/applications/_shared/nexus_factory.py",
    REPO_ROOT / "intergrax/applications/_shared/environment_wiring.py",
    REPO_ROOT / "intergrax/applications/_shared/harness_host_runtime.py",
)

BARE_CALL = re.compile(r"resolve_llm_adapter\s*\(\s*env\b")


def main() -> int:
    errors: list[str] = []
    for path in WIRING_FILES:
        if not path.is_file():
            errors.append(f"missing wiring module: {path.relative_to(REPO_ROOT)}")
            continue
        text = path.read_text(encoding="utf-8")
        if "resolve_environment_llm_adapter" not in text and BARE_CALL.search(text):
            errors.append(
                f"{path.relative_to(REPO_ROOT)}: use resolve_environment_llm_adapter() "
                "or pass routing_context/context_provider to resolve_llm_adapter()",
            )
        for match in BARE_CALL.finditer(text):
            errors.append(
                f"{path.relative_to(REPO_ROOT)}:{text.count(chr(10), 0, match.start()) + 1}: "
                "bare resolve_llm_adapter(env) is forbidden",
            )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("check_llm_routing_context_wiring: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
