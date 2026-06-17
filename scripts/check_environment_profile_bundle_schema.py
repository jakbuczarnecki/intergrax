#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""APP-EVOL-8.7 — ApplicationEnvironmentProfile bundle schema gate."""

from __future__ import annotations

import sys

from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    CapabilityBundle,
    CognitionBundle,
    EnvironmentExtensions,
    GovernanceBundle,
    HostMeta,
    IsolationBundle,
    SecurityEnvelope,
    TopologyBundle,
    bundle_normalized_payload,
    flatten_profile_dict,
    lift_flat_profile_dict,
)
from intergrax.applications.contracts.environment_profile.normalization import (
    BUNDLE_ROOT_KEYS,
)

from intergrax.integrations.registry.bootstrap import register_default_integrations


_BUNDLE_MODELS = (
    HostMeta,
    SecurityEnvelope,
    CapabilityBundle,
    CognitionBundle,
    GovernanceBundle,
    TopologyBundle,
    IsolationBundle,
    EnvironmentExtensions,
)


def _schema_forbids_extra(model: type) -> bool:
    schema = model.model_json_schema()
    return schema.get("additionalProperties") is False


def main() -> int:
    register_default_integrations(override=True)

    root_schema = ApplicationEnvironmentProfile.model_json_schema()
    properties = root_schema.get("properties", {})
    missing = sorted(BUNDLE_ROOT_KEYS - set(properties))
    if missing:
        print(
            f"ApplicationEnvironmentProfile schema missing bundle roots: {missing}",
            file=sys.stderr,
        )
        return 1

    for model in _BUNDLE_MODELS:
        if not _schema_forbids_extra(model):
            print(f"{model.__name__} schema must set additionalProperties=false", file=sys.stderr)
            return 1

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="bundle.schema.gate")
    flat = env.model_dump(mode="json")
    if any(key in flat for key in BUNDLE_ROOT_KEYS):
        print("spec_version 1.x wire dump must remain flat", file=sys.stderr)
        return 1

    nested = env.bundle_dump(mode="json")
    restored = ApplicationEnvironmentProfile.model_validate(nested)
    if restored.profile_id != env.profile_id:
        print("nested bundle round-trip failed", file=sys.stderr)
        return 1

    flat_lift = lift_flat_profile_dict(flat)
    flat_digest = bundle_normalized_payload(flat_lift)
    nested_digest = bundle_normalized_payload(nested)
    if flat_digest != nested_digest:
        print("flat vs nested bundle digest mismatch", file=sys.stderr)
        return 1

    _ = flatten_profile_dict(nested)

    print("OK: environment profile bundle schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
