#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""IDEAL-5.1 — pre-context policy hook wiring audit."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from intergrax.runtime.policy.pre_context_policy_audit import audit_pre_context_policy_wiring

    missing = audit_pre_context_policy_wiring(REPO_ROOT)
    if missing:
        print(f"missing pre-context policy markers: {missing}", file=sys.stderr)
        return 1
    print("OK: pre-context policy wiring markers present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
