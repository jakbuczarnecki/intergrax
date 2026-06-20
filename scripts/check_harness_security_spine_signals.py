#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Validate platform.security spine kinds and payload schema registration (Phase SEC-ENT-3)."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "agents", REPO_ROOT / "applications"):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from intergrax.core.security_bootstrap import bootstrap_security_providers, register_security_payload_schemas
from intergrax.runtime.events.event_kind_registry import get_event_kind_entry
from intergrax.runtime.events.spine_consolidation import get_platform_kind_entry
from intergrax.runtime.security.security_events import (
    KIND_DEFENSE_BLOCKED,
    KIND_ENCRYPTION_DENIED,
)


def main() -> int:
    register_security_payload_schemas()
    bootstrap_security_providers(discover_entry_points=False)
    for kind in (KIND_DEFENSE_BLOCKED, KIND_ENCRYPTION_DENIED):
        if get_platform_kind_entry(kind) is None:
            print(f"missing platform kind catalog entry: {kind}")
            return 1
        entry = get_event_kind_entry(kind)
        if entry is None:
            print(f"missing event_kind registry entry: {kind}")
            return 1
    print("harness security spine signals audit: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
