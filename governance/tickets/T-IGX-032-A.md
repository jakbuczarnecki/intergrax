# T-IGX-032-A - HIST03-F-015: strict production host composition crashes when it supplies both a KV store and a document store to skill execution pinning (PROD-Q P2-C corrective; owner SKILL-X; KV authoritative)

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the IIAC re-audit/re-activation batch.**

## Scope

`tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py` fails 6 of 7 with `ValueError: wire_skill_execution_pinning_store accepts kv_store or document_store, not both`. Q-11 confirmed a code defect introduced by SKILL-X Batch C `d1040f485`: strict composition always supplies both a sqlite KV store and an in-memory document store, `harness_host_runtime.py:423` forwards both to `resolve_skill_execution_pinning_wiring`, which forwards both to the strict leaf `wire_skill_execution_pinning_store`. The sibling effective-profile wiring in the same host runtime already resolves coexisting stores to KV in its resolver (`profile_resolution/wiring.py:68`) while its leaf stays strict.

Fix (recommended alternative A2): the pin RESOLVER passes the document store to the leaf only when no KV store exists, so KV is authoritative and the document store is the fallback. The leaf `wire_skill_execution_pinning_store` and its dual-backend rejection, the durability validator, the host runtime and all factories stay untouched. The P2-C test file is not edited or weakened.

Exact files: `intergrax/skills/skill_execution_pinning_wiring.py` (product, ~3 lines) and `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py` (new, RED-first). Alternatives (leaf accepts both; host runtime passes one store; strict composition supplies one store), blast radius of every caller, durability semantics, probes, test contract, rollback, residuals and Operator questions: [`plans/PLAN_T-IGX-032-A.md`](plans/PLAN_T-IGX-032-A.md).

No PASS, closure, FRZ promotion or stage-parent change is implied. PROD-Q P2-C is the victim and the re-verification gate, not the owner. Independent audit follows in a fresh session; implementer, auditor and plan approver are three different sessions.

## Plan digest

Plan digest: 2628d1a834d2fcf25f47c55b0d60355811898c9e60337306b557f152d7ba7446

## Approved

Approved: operator-delegate hist-sign-plans @ 2026-11-08T21:26:00+01:00 | Delegation: docs/project/maintainers/qualification/historical_certification_campaign/delegations/DELEG-HIST-PRODUCT-PLANS-2026-11-08.md@9bfa07fa97c5054b07ff1f1dc9799d9ca77471f5

Authority cited for the scope (given in chat and in the accepted decision records, not signature lines): DR-6 (Q-11 commissioned; ticket with an owner may follow), W1-PLAN step 3 (names the P2-C corrective with the KV store authoritative and SKILL-X as owner; a sequence only, approves no plan), DR-12 = B (HELD for ACT-N11 sequencing; the fix is necessarily a product-path change).

## Test contract

Phase 2 only, one pytest / ruff / uv process at a time, each under the heavy lock `D:/Projekty/intergrax/.tmp/session/hist-cert-campaign/locks/heavy.lock`, pytest's own exit code captured, logs under `.tmp/session/t-igx-032-a/`, always `uv run --extra dev-ci`:

```text
uv run --extra dev-ci pytest tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py -q   # RED before the product commit (3 failed with "not both", 5 passed); all passed after
uv run --extra dev-ci pytest tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py -q   # before (measured by Q-11): 6 failed, 1 passed; after: 7 passed; file byte-identical
uv run --extra dev-ci pytest <batch C pin test + p1_10 + batch D + batch E + effective_profile_persistence_wiring> -q   # one process; all passed, dual-backend leaf test untouched and green
negative probes (uncommitted, restored): revert the product line => exactly the 3 RED tests; invert precedence => KV tests red; drop the leaf guard => leaf tests red
bounded wider measurement (named files only, before and after, Operator to confirm): research factory e2e + 3 strict governed-contractor unit files
uv run --extra dev-ci ruff check <2 files> ; uv run --extra dev-ci ruff format --check <2 files>
git diff --name-only af323c005..HEAD    # only the two files + ticket/plan files
```

Informational and expected red (not to be fixed here): `verify_iiac_policy_activation --require-in-force` (ACT-N11, one product path). Exact commands, expected counts and stop rules are in section 7 of the plan.

## Out of scope

The leaf `wire_skill_execution_pinning_store` and its guard; the durability validator (hard-coded `is_durable`, plan residual R1); `harness_host_runtime.py`, the factories and `ProductionPlatformPersistence`; the P2-C test file; PROD-Q / SKILL-X records, roadmap and FRZ statuses; the six IIAC material governance paths and any IIAC record or activation; `T-IGX-030-B`, `T-IGX-030-C`, `T-IGX-031-A`, BND-03, `T-IGX-021-B/C`; PASS / closure claims; Docker / vendor / production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_approved
