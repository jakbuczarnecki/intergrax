# © Artur Czarnecki. All rights reserved.

"""Policy fragment versioning via ProfileVersionStore (Phase W-ADAPT-4.9)."""

from __future__ import annotations

from dataclasses import replace

from intergrax.runtime.adaptive.contracts import ProfileArtifactType, ProfileVersionRecord
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle


def apply_policy_fragment_version(
    bundle: RuntimePolicyBundle,
    version: ProfileVersionRecord | None,
) -> RuntimePolicyBundle:
    """Attach versioned policy fragment payload to a runtime policy bundle."""
    if version is None or version.artifact_type != ProfileArtifactType.POLICY_FRAGMENT:
        return bundle

    fragments = dict(bundle.domain_fragments)
    fragments["adaptive_policy_fragment"] = dict(version.artifact_payload)
    fragments["policy_fragment_version_id"] = version.version_id
    return replace(bundle, domain_fragments=fragments)
