"""P7-00 probe X8: does a non-strict profile sign boundary events with the public demo seed when no key is set?"""
import json
from intergrax.applications._shared.attestation_runtime_bridge import build_boundary_event_buffer
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.environment_profile.sub_profiles import ExecutionBoundaryExportProfile
from intergrax.core.security.security_configuration import SecurityConfigurationError

out = {}
for label, factory in (("lab_defaults", ApplicationEnvironmentProfile.lab_defaults), ("product_defaults", ApplicationEnvironmentProfile.product_defaults)):
    env = factory()
    env.execution_boundary_export_profile = ExecutionBoundaryExportProfile(enabled=True, host_signing_enabled=True)
    try:
        buf = build_boundary_event_buffer(env, environ={})
        out[label] = {"application_profile": str(env.application_profile), "execution_mode": str(env.execution_mode), "buffer_with_sealer": buf is not None}
    except SecurityConfigurationError as exc:
        out[label] = {"application_profile": str(env.application_profile), "execution_mode": str(env.execution_mode), "refused": exc.reason_code}
print(json.dumps(out, indent=2))
