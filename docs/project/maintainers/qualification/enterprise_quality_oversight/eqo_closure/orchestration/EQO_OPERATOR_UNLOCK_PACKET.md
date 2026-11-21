# EQO operator unlock packet

**Fetched:** 2026-11-21. Local `development` `1183221658cdba9d0c1e466b053b18d45c998f08`. `origin/development` `17fc79228eb0b46880ecad17ff52354622a81e9e`. Ahead 40, behind 0. Worktree `D:/Projekty/_wt/intergrax/af-dev-merge`. Clean. Not pushed. Not `EQO_CLOSED`. A4-006 and A4-027 are not formally settled.

## Publication

Product diff versus origin is seven paths: `applications/research_application/serving/fastapi_router.py`, `schemas.py`, `scripts/maintenance/check_runtime_event_tenant_propagation.py`, the two named unit tests, `governance/tickets/T-IGX-055-A.md`, `governance/tickets/T-IGX-055-B.md`. No secret-like strings in that diff. The rest of the 36-file delta is EQO bookkeeping. Reviewed behavior: missing and whitespace-only `tenant_id` fail closed; the gate lists `uaep_executor.py`. Pytest on the pre-publication SHA: A4-027 3 passed, A4-006 4 passed, integration 19 passed. That is `TECHNICAL_REVIEW_PASS` only.

Eleven unpublished commits carry `Co-authored-by: Cursor <cursoragent@cursor.com>`: `5cceebb69`, `537a06296`, `5e9496611`, `b02278643`, `d1da3b3c1`, `c600f79a6`, `c15588151`, `ffec97177`, `531b4e323`, `960205aaa`, `321957488`. Two commits lack `Ref:`: merge `6fa8f2955` and docs commit `d1da3b3c1`. History was not rewritten.

**OD-PUB (required before any further SHA-bound settlement).** Recommended: publish as-is. Do not rebase. From the development worktree:

```text
git fetch origin
git rev-list --left-right --count origin/development...HEAD
git push -u origin HEAD:eqo/recovery-publication
gh pr create --repo jakbuczarnecki/intergrax --base development --head eqo/recovery-publication --title "Publish EQO recovery fixes and bookkeeping" --body "Local development stack through 118322165. Technical review only. No EQO closure."
```

Do not push `development` directly. Do not merge a red required check. The alternate, not recommended, is an explicit rewrite authorization for the eleven trailers before push.

## Formal settlement still missing for A4-006 and A4-027

`TECHNICAL_REVIEW_PASS` does not settle `EDER-F-0115` / `T-IGX-055-A` or `EDER-F-0136` / `T-IGX-055-B`.

1. OD-PUB lands on `origin/development`.
2. Orchestrator re-runs the two named test files on that SHA.
3. An independent certifier, not the Composer implementer and not this technical review, records the WB02 / WB row against that SHA.
4. Nobody sets `CLOSED`, `FULLY_MET`, or FRZ-TEN-10 from these commits. A4-027 does not cover MCP `research-user` or API-key tenant `default`.

## One decision set

Accepting a row authorizes the package. Rejecting it leaves the blocker.

| ID | Block | Existing authority | Recommended resolution | Write set after acceptance | Acceptance test | Composer package |
|----|-------|--------------------|------------------------|----------------------------|-----------------|------------------|
| OD-040 | A4-040 direct `uvicorn.run` on reference apps | OD-EQO-04, open | Scoped exception for reference Tier-3 apps, with a reopen condition if one becomes a production entry. Not a hosting-engine migration. | `eqo_closure/` exception record only | Record cites the five entry files and the reopen condition | `EQO-PKG-A4-040-RECORD` |
| OD-042 | A4-042 Poland defaults in `intergrax/globals/settings.py` | OD-EQO-09, open | Neutral defaults (`en`, `UTC`); env override stays. Assign `intergrax/globals` as owner. | `intergrax/globals/settings.py` plus one unit test | Import-time defaults are not `pl` / `Europe/Warsaw` unless env sets them | `EQO-PKG-A4-042` |
| OD-031 | A4-031 UAEP skips assembly when context engine or adapter is None | T-IGX-043-A implementation go, open | Fail closed on that path. No silent skip. | `intergrax/runtime/nexus/uaep/uaep_executor.py` plus a negative test | Unset engine or adapter raises; the configured path stays covered | `EQO-PKG-A4-031` |
| OD-037 | A4-037 `human_store is None` returns | T-IGX-043-A implementation go, open | Fail closed outside an explicit lab opt-in. Do not invent a new store. | `intergrax/runtime/nexus/orchestration/human_response.py` and the loop call site that passes None, plus a negative test | Production construction without a store fails; lab opt-in still returns | `EQO-PKG-A4-037` |
| OD-034 | A4-034 five retry budgets, no owner sentence | `EQO-PKG-A4-034/IMPLEMENTATION_BLOCKED.md`, FRZ-REL-02 | Sanction, do not delete: inline budget is the LLM default; `LocalProviderRetryBudget` stays unwired; `StepRetryBudget` stays the step-layer owner and unwired. | `eqo_closure/` sanction record only. No `intergrax/` edit. No FRZ edit. | Record quotes the three mechanisms and forbids deletion | `EQO-PKG-A4-034-RECORD` |
| OD-006 | A6-006 semantic skip | `EQO-PKG-A6-006/EXACT_BLOCKER.md`, ENT-AUDIT-X contract | Accept `ENT_AUDIT_X_SEMANTIC` skip as by-design. No gate retune. | `eqo_closure/` acceptance record only | Record cites the contract; the skip remains | `EQO-PKG-A6-006-RECORD` |
| OD-034N | 34 `NO_NAMED_TEST` rows | `EQO_REPLAY_REST.json` `remainder_batch` | Do not invent tests. They stay blocked until FCS / EQO-REPLAY-FCS. | None | None until that gate | none |

## Who acts

Operator only: OD-PUB and the seven rows above. Orchestrator after acceptance, without another continue order: spawn the authorized Composer packages on disjoint worktrees, review, test one `uv` at a time, merge locally, do not push. Doc-only packages (`040`, `034`, `006`) may run beside code packages. `031`, `037`, and `042` are disjoint code and may run together. Orchestrator does not edit FRZ, IIAC, `register.jsonl`, or set `CLOSED`.

Already inside the mandate and done: technical review, the two fixes, named-test replay. Nothing else in Q1–Q5 is READY before these decisions.

## Next waves

1. **PUB** — operator procedure above. No Composer.
2. **F2 unlock** — only the rows accepted above. Code packages first, then their tests, then local merge.
3. **F3 / F4** — B-2 owner PRs and the 34-row replay stay closed until W4-03 / FCS. No wave is started for them by this packet.

Resume condition: OD-PUB result SHA, plus which OD rows were accepted. Rejected rows stay blocked.
