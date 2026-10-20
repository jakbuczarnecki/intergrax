# © Artur Czarnecki. All rights reserved.

"""Fail-closed CONFIGURED admission gate for profile v1 (not TARGET-VERIFIED)."""

from __future__ import annotations

import sys

from intergrax.applications._shared.single_node_production_profile_v1 import (
    SingleNodeProductionProfileV1ConfigurationError,
)
from governed_contractor_application.host.single_node_profile_v1_strict_startup import (
    resolve_strict_production_storage_from_single_node_production_profile_v1,
)


def main(argv: list[str] | None = None) -> int:
    _ = argv
    try:
        _, _, hold = resolve_strict_production_storage_from_single_node_production_profile_v1(
            acquire_cross_process_writer=False,
        )
        hold.release()
    except SingleNodeProductionProfileV1ConfigurationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
