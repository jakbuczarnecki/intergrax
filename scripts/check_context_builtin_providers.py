#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail when shipped builtin context providers still use empty stub collect (CE-PROV-GATE)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from intergrax.context.providers.builtin import (
        WIRED_BUILTIN_COLLECTOR_IDS,
        _COLLECT_OVERRIDES,
    )

    missing = sorted(
        provider_id
        for provider_id in WIRED_BUILTIN_COLLECTOR_IDS
        if provider_id != "builtin.session_history_semantic"
        and provider_id not in _COLLECT_OVERRIDES
    )
    if missing:
        print("builtin context providers missing live collect() wiring:")
        for provider_id in missing:
            print(f"  - {provider_id}")
        return 1
    print("context builtin providers: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
