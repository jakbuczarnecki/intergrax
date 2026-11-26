# Integrax Intelligence — delivery plan index (multi-layer feature)

**Architecture hub (SSOT):** [`architecture/INTEGRAX_INTELLIGENCE.md`](../architecture/INTEGRAX_INTELLIGENCE.md)  
**Proposed program ID:** `INTELLIGENCE-X`  
**Status:** **PROPOSED — NOT RATIFIED / NOT IMPLEMENTATION AUTHORITY**

Do not duplicate the 69-package WBS or acceptance matrix here. The architecture hub owns chapters 10–12 (waves, workloads, gates, acceptance criteria).

## Strategic placement (proposed)

```text
ARCH-FREEZE + required Enterprise Campaign acceptance
        → INTELLIGENCE-X (W0–W9)
        → GATE-INTELLIGENCE-CONSUME
        → Scenario Factory productive implementation (SFX-P5+) and subsequent consumers
```

**Consistency rules (proposed, pending operator ratification):**

- INTELLIGENCE-X is **not** a prerequisite for ARCH-FREEZE closure.
- `GATE-INTELLIGENCE-CONSUME` is proposed as an additional SFX-P5+ entry condition; approved `SFX-MASTER-PLAN.md` digest amendments remain **DEFERRED_PENDING_AUTHORITY**.
- SEF-X shared-intelligence reuse and OD-SEF-02 Option 3 preserved unless explicitly amended in `ASF-SEF-MASTER-PLAN.md` (deferred).

## Proposed follow-on artifacts (not created in this ticket)

| Path | Role |
|------|------|
| `governance/tickets/plans/INTELLIGENCE-X-MASTER-PLAN.md` | Executable WBS SSOT once ratified (until then: hub §10) |
| `docs/project/maintainers/qualification/intelligence_x/INTELLIGENCE_X_INDEX.md` | Evidence dossier index |
| `docs/project/maintainers/qualification/intelligence_x/INTELLIGENCE_X_ACCEPTANCE_MATRIX.md` | Acceptance evidence mapping |
| `docs/project/maintainers/qualification/intelligence_x/INTELLIGENCE_X_OWNERSHIP_MATRIX.md` | Concern-level reuse mapping |

## Gates (proposed — see architecture hub §12)

| Gate | Summary |
|------|---------|
| `GATE-II-ENTRY` | Post-freeze program entry |
| `GATE-II-ARCH` | W0–W1 architecture eligibility |
| `GATE-II-CORE` | W2–W4 integrated goal/cognition/team behaviors |
| `GATE-II-ADAPTIVE` | W5–W7 adaptive tool/context/recovery/verification behaviors |
| `GATE-II-ENTERPRISE` | W8 enterprise scale and isolation |
| `GATE-INTELLIGENCE-CONSUME` | W9 + mandatory acceptance rows — consumer adoption |