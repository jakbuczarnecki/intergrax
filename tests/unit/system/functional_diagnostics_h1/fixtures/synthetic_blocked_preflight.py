# © Artur Czarnecki. All rights reserved.

"""Controlled synthetic blocked preflight for H1 classification."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    credential = os.environ.get("H1_SYNTHETIC_REQUIRED_CREDENTIAL", "").strip()
    if not credential:
        print(json.dumps({"verdict": "BLOCKED", "reason": "missing_required_credential"}))
        return 2
    print(json.dumps({"verdict": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
