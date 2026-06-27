#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Regression gate: CE OTel span registry (CE-9.2)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    sys.path.insert(0, str(_REPO))
    from intergrax.context.tracking.context_spans import CE_OTEL_SPAN_NAMES

    source = (_REPO / "intergrax/runtime/nexus/context/context_engine.py").read_text(encoding="utf-8")
    errors: list[str] = []
    for span_name in CE_OTEL_SPAN_NAMES:
        pattern = re.compile(rf'context_span\(\s*["\']{re.escape(span_name)}["\']')
        if not pattern.search(source):
            errors.append(f"context_engine.py missing span {span_name!r}")
    if errors:
        for err in errors:
            print(f"check_context_otel_span_registry: FAIL — {err}", file=sys.stderr)
        return 1
    print("check_context_otel_span_registry: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
