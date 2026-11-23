# SFX R1 → R2 Delta and Reuse Matrix

**HEAD:** `b4146cbfab9858c058fc44df2ad43d8670d4925d`

## Summary

| Category | R1 | R2 additive |
| --- | --- | --- |
| Architecture placement | Option B | **Unchanged** |
| Lifecycle UX steps | 11 linear | + Mission cockpit, iteration timeline, champion compare |
| Engineering loop | Single build + human Improve | + Goal loop, candidate iterations, autonomous driver (grant-gated) |
| Contracts | 17 `sfx.*.v1` | +6 PROPOSED schemas (see contract catalog) |
| Requirements | REQ-SFX-001..054 | + REQ-R2-001..018 (proposal JSON) |
| WBS | SFX-P0..P13 | R2-W3..W8 overlay units (map to P5..P13) |

## Reuse vs new (platform)

| Capability | R1 owner | R2 reuse |
| --- | --- | --- |
| Scenario execution | Intergrax Runtime / NexusLoop | **100% reuse** |
| Engineering jobs | SEF-X EEA | **100% reuse**; optional grant extension |
| Persistence | CW, Session, LR, Evidence | **100% reuse** — new records are artifact types |
| Approvals | DecisionBinding / HITL | **Reuse** + envelope grant ref |
| Policy | PolicyEngine | **Reuse** — no SFX policy authority |
| Eval | EvalRunner / proof runner | **Reuse** — SFX verdict view only |
| Tier-3 doors | Studio/API/MCP/CLI pattern | **Extend** op catalog |

## R2 requirements coverage (high level)

| R2 need | Existing R1 | Gap / amendment |
| --- | --- | --- |
| Measurable business goals | Partial in CON-01/03 | R2-W3-01 |
| Source intake manifest | Discovery only | R2-W3-02, SFX-GAP-R2-02 |
| Autonomy envelope | Approval ref only | R2-W3-03, SFX-GAP-R2-01 |
| Multi-candidate iterations | Improve step manual | R2-W3-04, W4-05, SFX-GAP-R2-03 |
| Autonomous driver | Absent | R2-W5-03, SFX-GAP-R2-04 |
| Business oracle eval | Test plan exists | R2-W5-01, SFX-GAP-R2-05 |
| HO-2 multi-submit | Forbidden | SEF amendment, C-SEF-R2-01 |
| Mission metrics AUTO-* | M-1..M-9 only | Metrics appendix + SFX-GAP-R2-06 |
| Customer self-service portal | Not in scope | Documented later phase only |

Legend: **REUSE** · **EXTEND** · **MISSING** (genuine new SFX product logic).