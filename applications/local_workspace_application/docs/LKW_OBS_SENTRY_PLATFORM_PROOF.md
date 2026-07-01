# LKW-OBS-SENTRY — Sentry Error-Monitoring Platform Proof

Status: **Planned / Platform-reusable**

This document records a Local Knowledge Workspace driven platform proof candidate. It is intentionally scoped as a planning artifact only. It does not implement Sentry, add dependencies, add environment variables, or change runtime behavior.

## Why this belongs in the LKW plan

LKW is the proof workload for Intergrax platform capabilities. When LKW exposes a reusable observability or diagnostics need, the implementation should move to shared platform/provider code while LKW keeps only deployment-specific configuration and live proof responsibilities.

Sentry is therefore tracked here as an LKW-discovered platform proof, not as a local workspace product feature.

## Goal

Prove that Intergrax can support a provider-owned Sentry error-monitoring integration for safe exception capture and issue triage from an agent application workload.

Sentry should complement Elasticsearch/Kibana timeline observability. It must not replace structured observability export or the Elasticsearch/OpenSearch observability backend proof.

## Intended ownership boundary

```text
LKW / Tier-3 application
→ deployment env and live proof workload only
→ typed runtime/operator config
→ provider-owned Sentry integration
→ Sentry error monitoring / issue triage backend
```

LKW must not call the Sentry SDK directly and must not implement Sentry-specific capture behavior.

## Expected safe metadata only

A future Sentry proof may attach safe tags/context such as:

```text
run_id
task_id
agent_id
capability
environment
release
operation
error_reason
backend_id
```

The integration must not send:

```text
prompt
answer
chunks
document text
tool args
raw observability envelope
raw payload
secrets
tokens
absolute payload paths
user file content
```

## Proposed platform track

Detailed implementation should be tracked in `docs/plan/OBSERVABILITY.md`, for example as:

```text
OBS-SENTRY — Planned
```

Suggested decomposition:

```text
OBS-SENTRY-0 — design note / scope boundary
OBS-SENTRY-1 — provider config + no-op behavior
OBS-SENTRY-2 — provider-owned sentry-sdk adapter
OBS-SENTRY-3 — runtime typed config wiring
OBS-SENTRY-4 — LKW env wiring
OBS-SENTRY-5 — safe exception capture proof
OBS-SENTRY-6 — live Sentry proof
```

## Suggested future LKW wave entry

```text
LKW-OBS-SENTRY — Sentry error-monitoring platform proof
Depends: OBS-VENDOR-6C live proof / OBS-SENTRY platform design
Status: Planned / Platform-reusable
Priority: Medium
```

## Out of scope until explicitly scheduled

- Installing or importing `sentry-sdk`.
- Adding Sentry env vars to LKW.
- Capturing live exceptions.
- Sending performance traces or profiling data.
- Sending PII.
- Alert routing.
- Replacing Elasticsearch/Kibana timeline observability.

## Acceptance for future proof

A future implementation may be considered proven only when:

- Sentry provider behavior is owned by platform/provider code.
- LKW owns only env/deployment configuration and proof workload.
- `send_default_pii` or equivalent behavior is disabled by default.
- A controlled LKW failure creates a Sentry issue/event with safe tags only.
- No prompt, chunks, tool args, secrets, raw documents, payload content, or absolute payload paths are sent.
- The proof is documented with concrete evidence in the relevant LKW and platform observability plans.
