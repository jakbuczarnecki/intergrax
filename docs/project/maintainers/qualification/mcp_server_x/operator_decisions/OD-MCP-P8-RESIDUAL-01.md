# OD-MCP-P8-RESIDUAL-01 — disposition of R-P8-01 and OPEN-BP-01 after the MCP-SERVER-X scoped closure

| Field | Value |
| --- | --- |
| **Decision ID** | OD-MCP-P8-RESIDUAL-01 |
| **Status** | **RECORDED (operator chat instructions; not signed `Authorized:` lines). Part 1 accepted; part 2 (OPEN-BP-01) DEFERRED by the operator** |
| **Related** | R-P8-01, OPEN-BP-01, `OD-PDC-SEQ-01`, `OD-PDC-IIAC-SCOPE`, `OD-MCP-P8-CLOSE-01` |
| **Policy version** | IIAC-2026-11-05.1 |

## Question

Which path does the operator take for the two open-items-list entries that stayed with the operator after the closure: (1) enabling required status checks on `development` (OPEN-BP-01) and (2) the fail-closed composition fix for R-P8-01 (a host mounted without harness auth trusts the client-asserted tenant and user)?

## Operator record

Operator chat, 2026-11-12. Polish original: "Tak, oba". English rendering: "Yes, both". It answered the question "Do you enable required checks on `development` and combine R-P8-01 with the PDC-P7-00 rebind?". Recorded here as the operator's instruction; no agent-composed timestamp and no signature line.

## Disposition

1. **R-P8-01 is bundled into the PDC-P7-00 ACT-N11 activation.**
   - The fix (fail-closed composition when harness auth is absent) is a product change under `intergrax/`, so it triggers ACT-N11.
   - It is **not** implemented now and **not** as a separate rebind. It is implemented as its own ticketed batch (ticket-first, plan digest, `Approved:` line) in the PDC-X integration window, so one independent delta audit and one two-phase rebind at `INTEGRATED_BASELINE_SHA` (PDC-P7-00) cover it together with the PDC-X domain batches (OD-PDC-SEQ-01, OD-PDC-IIAC-SCOPE).
   - Until that batch is merged and certified, R-P8-01 stays an **open scoped residual** exactly as stated in the P8 closure documents. Nothing is waived and nothing is closed by this record.
2. **OPEN-BP-01 is DEFERRED by the operator and stays an accepted limitation.**
   - Operator chat, 2026-11-12, Polish original: "Moja decyzja jest taka, że nie będziemy na razie modyfikować rulesetu, ponieważ rozmawiamy o gałęzi development, która nie jest krytyczna. Krytyczną gałęzią jest gałąź main, dlatego teraz nie będziemy się tym zajmować." English rendering: "My decision is that we will not modify the ruleset for now, because we are talking about the development branch, which is not critical. The critical branch is main, so we will not deal with this now."
   - Read-back on 2026-11-12 (`gh api repos/jakbuczarnecki/intergrax/rules/branches/development`): the effective rules on `development` are `deletion` and `non_fast_forward` only (ruleset `26771061`, last updated 2026-11-07); `main` (ruleset `20844442`) carries `pull_request` and `required_status_checks`. No change was made in GitHub.
   - Consequence, stated and accepted: merges into `development` are not blocked by CI. The agent rule "merge only after the five required checks finished green" is the only control; manual merges before the checks finish remain possible and are recorded as deviations (P8-PROC-01..03 and earlier cases), not waived.
   - If the operator later wants the control, the five contexts to add to ruleset `26771061` are `ci-smoke`, `governance-fast`, `evol-x-b4-preflight`, `ent-audit-fast`, `ent-audit-pr-gate`. The legacy script `scripts/dev/apply_github_branch_protection.ps1` requires only `ci-smoke` and uses the branch-protection API, not the ruleset, so it is not the recommended path.

## Explicit non-actions

No product-path change, no IIAC activation record change, no FRZ change, no ticket opened or approved by this record, no change to the PDC-X pull requests #475, #476 and #478, no change to any GitHub ruleset.
