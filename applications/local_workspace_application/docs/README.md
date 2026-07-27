# Local Knowledge Workspace — documentation index

Application-local documentation for the Tier-3 `local_workspace_application` host.

Only [`../README.md`](../README.md) lives at the application root. All other LKW Markdown docs belong under this directory.

## Canonical docs

| Document | Purpose |
|----------|---------|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Canonical architecture (includes deployment-neutral storage and tenancy) |
| [`ARCHITECTURE_HARDENING.md`](ARCHITECTURE_HARDENING.md) | Hardening decisions |
| [`KNOWLEDGE_INTAKE_DISCOVERY.md`](KNOWLEDGE_INTAKE_DISCOVERY.md) | Channel-neutral intake, upload, source, async operation and notification contract |
| [`PLATFORM_PROOF_LOOP.md`](PLATFORM_PROOF_LOOP.md) | Product/platform propagation checklist |
| [`USER_JOURNEY.md`](USER_JOURNEY.md) | User-facing product journey |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Implementation wave plan |
| [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) | Build/run/deploy instructions |
| [`LKW_1_LIVE_VERIFICATION.md`](LKW_1_LIVE_VERIFICATION.md) | Current live verification status |
| [`../README.md#developer-first-run`](../README.md#developer-first-run) | Developer first-run path (in README) |

## Local history and decisions

| Path | Purpose |
|------|---------|
| [`journal/`](journal/) | Application-local implementation history |
| [`adr/`](adr/) | Local architecture decision records |

## Global vs application journals

LKW-specific implementation history belongs in [`journal/`](journal/), **not** in the platform-wide [`docs/implementation-journal/`](../../../docs/implementation-journal/).

Use the global implementation journal only for platform-wide milestones that affect multiple tiers or domains.
