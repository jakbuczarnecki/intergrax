# HARNESS-01-ADR2 — Architecture Decision Acceptance Record

| Field | Value |
|-------|-------|
| **Status** | **ACCEPTED** (architecture freeze only) |
| **ADR** | [`ADR-HARNESS-001`](../../technical/adr/entries/2026-09-20/ADR-HARNESS-001.md) |
| **Baseline HEAD** | `3dc68c32e2fdd7710d47dba6135e7a2bba964851` |
| **R5 block SHA** | `358593fd6324cf4682eee9092ae0efac62ea2a2d` |
| **Production code** | **Unchanged** |

## Verdict

```text
HARNESS-01-ADR2: PASS — ADR ACCEPTED
Resolved by HARNESS-01-ADR2 / ADR-HARNESS-001.
```

## Frozen summary

- EE internal Nexus zones: `intergrax/runtime/execution/**` + `intergrax/runtime/nexus/**` only
- Public host run entry: `HostTaskExecutionPort`
- Composition target: `build_execution_engine(...)` (not implemented in ADR2)
- Public tool pattern ABI: `intergrax.tools.invocation_pattern`
- Final gate classes: `EE_INTERNAL` | `TEST_ONLY` | `VIOLATION` (no final `DEBT`)
- HARNESS-02 lifecycle authority: unchanged

## Next

```text
HARNESS-01-R5 — resume implementation under ADR2
```

Independent audit of ADR2 against current GitHub code/docs is required before R5 resumes.
