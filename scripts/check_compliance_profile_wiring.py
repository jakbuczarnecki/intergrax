#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-5.2 — compliance profile templates per regulated domain class."""

from __future__ import annotations

import sys

from intergrax.applications._shared.compliance_profile_wiring import resolve_compliance_profile_wiring
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile


def main() -> int:
    wiring = resolve_compliance_profile_wiring(ApplicationEnvironmentProfile.product_defaults())
    if not wiring.enabled:
        print("product host must enable compliance profile templates", file=sys.stderr)
        return 1
    if wiring.template is None or not wiring.domain_fragments.get("compliance_profile"):
        print("compliance domain fragments missing", file=sys.stderr)
        return 1
    print(f"OK: compliance profile wiring ({wiring.template.domain_class.value})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
