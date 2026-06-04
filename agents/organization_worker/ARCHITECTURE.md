# Organization worker agent — architecture

## Purpose

Long-running / HITL-oriented harness agent for vendor reporting (`org.vendor_report`).

## Capabilities

- `org.vendor_report`

## Runtime

- UAEP agent with checkpoint-friendly steps
- Stub LLM adapter in agent module for offline gate tests (no `testing_support` import from production path)

## Registration

- Optional lab roster entry via `LabApplicationSettings`

## Related platform features

- Human-in-the-loop escalation (see `AGENT_CREATION_GUIDE.md` Appendix A)
- Checkpoint store wired by Tier-3 `build_harness_host_runtime`
