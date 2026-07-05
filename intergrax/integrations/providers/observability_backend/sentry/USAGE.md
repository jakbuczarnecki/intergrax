# Sentry (sentry)

Category: `observability_backend`

## Single public entrypoint

- **`SentryObservabilityIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `SentryObservabilityIntegration`.
- Contract factory: `create_sentry_observability_integration()`.

## Provider-level export (OBS-SENTRY-1)

Build transport and integration explicitly — registry construction does not open the Sentry SDK.

```python
from intergrax.integrations.providers.observability_backend.sentry import (
    create_sentry_observability_integration,
    create_sentry_observability_transport,
)
from intergrax.runtime.observability.export_boundary import ExportRecordKind, ObservabilityExportEnvelope
from intergrax.runtime.observability.export_policy import ObservabilityExportPolicy, apply_observability_export_policy

transport = create_sentry_observability_transport(
    dsn="https://<key>@sentry.io/<project>",
    environment="staging",
)
integration = create_sentry_observability_integration(transport=transport, enabled=True)

envelope = ObservabilityExportEnvelope(
    record_kind=ExportRecordKind.PROBLEM_SIGNAL,
    problem_kind="example.failure",
    problem_severity="error",
    problem_error_code="EXAMPLE_FAILURE",
    run_id="run-1",
)
policy = apply_observability_export_policy(envelope, ObservabilityExportPolicy(enabled=True))
if policy.exported and policy.envelope is not None:
    await integration.export(policy.envelope)
```

DSN and SDK settings belong to provider config/env (`INTERGRAX_SENTRY_DSN`, etc.) — never to `ObservabilityVendorPayload`.

**Deferred:** LKW endpoint proof, docker compose, and live Sentry operator docs.
