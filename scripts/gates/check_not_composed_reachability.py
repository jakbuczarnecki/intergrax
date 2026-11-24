#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""T-AF2-RW-GATES: NOT_COMPOSED trust admission cannot pass through when enforcement is required."""

from __future__ import annotations

import argparse
import sys

from intergrax.applications._shared.deploy_activate_pipeline import (
    DeployActivatePipelineContext,
    RevisionTrustAdmissionState,
    _resolve_revision_trust_admission,
    assert_composed_revision_trust_admission,
)


def _probe_not_composed_raises() -> None:
    """Exercise the NOT_COMPOSED resolution path with enforcement enabled (no live deploy)."""

    class _Composition:
        agent_capability_runtime = None

    ctx = DeployActivatePipelineContext(
        composition=_Composition(),  # type: ignore[arg-type]
        services=object(),  # type: ignore[arg-type]
        mutation_authorization_boundary=object(),  # type: ignore[arg-type]
        environment_tenant_resolver=object(),  # type: ignore[arg-type]
        require_composed_trust_admission=True,
    )
    _hook, state = _resolve_revision_trust_admission(ctx)
    assert state is RevisionTrustAdmissionState.NOT_COMPOSED
    assert_composed_revision_trust_admission(ctx, state)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Run in-repo NOT_COMPOSED probe (raises on pass-through when enforcement is required)",
    )
    args = parser.parse_args()
    if not args.probe:
        print("DENIED: specify --probe to exercise NOT_COMPOSED reachability", file=sys.stderr)
        return 1
    try:
        _probe_not_composed_raises()
    except Exception as exc:  # noqa: BLE001
        print(f"OK: NOT_COMPOSED reachability refuses pass-through ({type(exc).__name__})")
        return 0
    print("DENIED: NOT_COMPOSED path passed through without raising", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
