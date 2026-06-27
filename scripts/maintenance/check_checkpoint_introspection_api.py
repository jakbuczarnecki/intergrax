#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-8.2 — checkpoint introspection API on product hosts."""

from __future__ import annotations

import sys

from intergrax.applications._shared.checkpoint_introspection_routes import (
    create_checkpoint_introspection_router,
)
from intergrax.applications._shared.checkpoint_introspection_wiring import (
    resolve_checkpoint_introspection_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.runtime.long_running.store import SQLiteTaskCheckpointStore


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    wiring = resolve_checkpoint_introspection_wiring(env)
    if not wiring.enabled:
        print("product host must enable checkpoint introspection", file=sys.stderr)
        return 1

    store = SQLiteTaskCheckpointStore(db_path=":memory:")
    router = create_checkpoint_introspection_router(store, enabled=True, prefix=wiring.route_prefix)
    paths = [route.path for route in router.routes]
    if not any(path.endswith("/checkpoints") for path in paths):
        print("checkpoint introspection router missing /checkpoints route", file=sys.stderr)
        return 1

    print("OK: checkpoint introspection API")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
