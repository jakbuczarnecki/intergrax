# Plan hub satellites

**Parent directory:** [`../`](..) — domain plan hubs (`<DOMAIN>.md`)

Load **one** satellite per session when a task or audit gap ID requires it.

| Pattern | When to load |
|---------|----------------|
| `<DOMAIN>_06_closed_queues.md` | Re-validating closed §6.1/§6.2 queues |
| `<DOMAIN>_master_registers.md` | Gap ID in ORCH/FLOW/TS/… master register |
| `<DOMAIN>_audit_history.md` | LC closeout / historical audit narrative |
| `<DOMAIN>_phase_closeout.md` | Phase V-REM, FAUDIT-32, INT closeout bodies |
| `<DOMAIN>_appendices.md` | Appendix traceability B–N |
| `PLATFORM_FOUNDATION_*` | Platform-wide registers (shared canonical source) |

**Regenerate splits:** `uv run python scripts/docs/split_domain_plan.py [DOMAIN ...]`

**Architecture satellites:** [`../architecture/satellites/`](../architecture/satellites) — `uv run python scripts/docs/split_domain_architecture.py [DOMAIN ...]`

**CI gates:** `check_plan_hub_size.py` · `check_arch_hub_size.py` · `check_token_generator_freshness.py`

**Audit compact context:** [`../guides/audit_slices/`](../guides/audit_slices)
