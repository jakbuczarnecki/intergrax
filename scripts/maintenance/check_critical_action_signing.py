#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-4.1 — cryptographic signing / audit-protect for critical actions (fail closed)."""

from __future__ import annotations

import secrets
import sys

from intergrax.applications._shared.critical_action_signing_wiring import (
    resolve_critical_action_signing_wiring,
)
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.core.security.security_configuration import (
    SIGNING_SECRET_MISSING,
    SecurityConfigurationError,
)


def main() -> int:
    env = ApplicationEnvironmentProfile.product_defaults()
    secret_env = env.identity_profile.critical_action_signing_secret_env

    # PDC-D6-R008: a product host without a signing secret must fail closed (no literal fallback).
    try:
        resolve_critical_action_signing_wiring(env, environ={})
    except SecurityConfigurationError as exc:
        if exc.reason_code != SIGNING_SECRET_MISSING:
            print(f"unexpected configuration error: {exc.reason_code}", file=sys.stderr)
            return 1
    else:
        print("product host without a signing secret must fail closed", file=sys.stderr)
        return 1

    wiring = resolve_critical_action_signing_wiring(env, environ={secret_env: secrets.token_hex(32)})
    if not wiring.enabled:
        print("product host must enable critical action signing", file=sys.stderr)
        return 1
    if wiring.bootstrap_signature is None:
        print("critical action bootstrap signature missing", file=sys.stderr)
        return 1
    print(f"OK: critical action signing ({wiring.bootstrap_signature.action_kind.value}); missing secret fails closed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
