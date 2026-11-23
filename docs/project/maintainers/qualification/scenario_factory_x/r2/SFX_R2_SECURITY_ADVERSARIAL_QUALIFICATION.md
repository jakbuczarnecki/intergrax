# SFX R2 — Security and Adversarial Qualification Plan (PROPOSED)

Extends R1 SFX-P9-05 scope for mission/autonomy surfaces.

## Threat categories

| ID | Threat | Control | Test family |
| --- | --- | --- | --- |
| T-R2-01 | Grant scope expansion via prompt injection in discovery | Untrusted labels; fail-closed spec | `adversarial/injection_mission` |
| T-R2-02 | Auto-submit without grant | HO-2 + SEF validation | `handoff/grant_required` |
| T-R2-03 | Budget bypass via parallel candidates | `max_parallel` + cost accounting | `adversarial/budget_parallel` |
| T-R2-04 | Cross-tenant mission/candidate read | tenant_id on all records | P5-05 crossover |
| T-R2-05 | Self-certification of sandbox success | SFX-INV-01, verdict separation | `eval/business_vs_code_pass` |
| T-R2-06 | Egress from L2 demo | deny_egress + SEF lease | P9-04 |
| T-R2-07 | Upload malware / zip bomb | scan quarantine | source intake |
| T-R2-08 | Revoked grant in-flight job | SEF cancel + idempotency | R2-W4-08 |

## Stop conditions (security)

Any T-R2-* exploitation → `ESCALATED`, champion frozen, grant revoked.

## Qualification gates

Runs under `tests/qualification/scenario_factory_x/` — no new global authority. GATE-SFX-FULL requires full suite green at pinned SHA.