# PROPOSED (non-binding) — M-09H human drill automation substitution

**Status:** **PROPOSED FOR OPERATOR CONSIDERATION** — **NOT APPROVED**, **NOT SIGNED**, **NOT APPLIED** to FRZ / P03 / B6 acceptance.

**Trigger:** Operator preference to run E2 diagnostics without an interactive human evaluator session. Automated batch executed 2026-11-01; see [EVOL_X_E2_AUTOMATED_QUALIFICATION_RECORD.md](./EVOL_X_E2_AUTOMATED_QUALIFICATION_RECORD.md).

## Requirement substitution matrix

| Original requirement | Human normative? | Proposed automated equivalent | What automation proves | Human-only property lost | Operator decision | Independent auditor | EVIDENCE_FINAL / B6 / FRZ |
|----------------------|------------------|--------------------------------|------------------------|--------------------------|-------------------|---------------------|---------------------------|
| C1–C3 qualified tool + cause identity @ TES | Partial (tool yes; attribution no) | Batch runner @ TES + registry oracle | Deterministic cause digest, input SHA, tool SLA, diagnostic integrity | — | **Executed** (diagnostic scope) | N/A for tool-only | Does **not** close M-09H |
| `attribution_seconds` median ≤900 s (OD-EVOL-08-M1) | **Yes** | *None in current authorities* | — | Human comprehension under time pressure; blind diagnosis | **REQUIRED** to waive or replace via new OD | **REQUIRED** if waiver adopted | Without waiver: FINAL **blocked** |
| `evol_m09_human_drill.v2.json` + worksheet | **Yes** | `evol_m09_automated_diagnostic_qualification.v1` (separate schema) | Machine metrics + digests | Signed human worksheet chain | **REQUIRED** to accept alternate schema for FINAL | B6 package review | P03 Mode A rejects v2 without human fields |
| AC-M09H-01a registry blindness | **Yes** | Not substitutable by agent | — | Independent human attestation | **REQUIRED** | Auditor verifies attestation | FINAL `adversarial_control_passed` stays false |
| AC-M09H-01b honeytoken resistance | **Yes** | Partial: registry decoy separation (AC-AUTO-06) | Decoys ∉ canonical registry | Post-seal human confirmation | **REQUIRED** for full AC-M09H-01 | Auditor | Human leg OPEN |
| AC-M09H-01c sealed timing integrity | **Yes** | Not substitutable | — | Pre-oracle human timestamps | **REQUIRED** | Auditor | OPEN |
| EVOL-M-09H → 3/3 MEASURED in `evol_metrics.v1.json` | **Yes** (FINAL) | **Forbidden** without human policy change | — | Metric semantics | **REQUIRED** | B6 | FRZ-EVOL-08 scope |
| B6 independent audit (OD-EVOL-11) | **Yes** | No agent self-audit | — | Independence | Auditor naming **REQUIRED** | **REQUIRED** | B6 **PENDING** |

## Authority read (Batch B)

- **OD-EVOL-08-M1 (OPERATOR APPROVED):** human measurements only from human evaluator; agent/synthetic timings forbidden for `attribution_seconds`.
- **EVOL_X_SHA_LADDER / B6 package:** FINAL adds `evol_m09_human_drill.v2.json`; TECH explicitly **M-09H NOT_MEASURED**.
- **EVOL_X_EXPERIMENT_AND_METRIC_SPEC:** M-09H dual SLA binds human drill v2 artifact.
- **No** canonical document authorizes replacing M-09H human PASS with automated diagnostic PASS for FINAL or EVOL-X closure.

## Proposed operator decision (draft — not ratified)

> «Accept **`evol_m09_automated_diagnostic_qualification.v1`** as **supplemental** evidence for cause-identity and tool-path integrity only. **Do not** map it to `EVOL-M-09H` MEASURED or `evol_m09_human_drill.v2` without a **new signed OD** and B6 acceptance of amended FINAL contract.»

**If rejected:** retain automated record as diagnostic-only; human drill path unchanged.

**Minimum path to EVOL-X CLOSED (unchanged):** genuine human M-09H drill + AC-M09H-01 + FINAL evidence commit + B6 acceptance + remaining FRZ/OD items per master plan.
