# T-IGX-034-I — PRE-ENT Wave 5: enterprise certification final matrix + roadmap §5 ledger row

## Scope

Wave 5 of campaign `pre_ent_enterprise_certification`: publish an honest predecessor disposition matrix and update campaign SSOT (closure path, blocker matrix, campaign progress, parent checkpoint). Optional single non-closing row in [`PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §5 Closure evidence ledger pointing at the final matrix and publication SHA.

**In scope (docs/governance only):**

- Create [`PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_ENTERPRISE_CERTIFICATION_FINAL_MATRIX.md).
- Update [`PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_WAVE4_TO_6_CLOSURE_PATH.md), [`PRE_ENT_WAVE4_BLOCKER_MATRIX.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_WAVE4_BLOCKER_MATRIX.md), [`PRE_ENT_CAMPAIGN_PROGRESS.md`](../../docs/project/maintainers/qualification/pre_ent_enterprise_certification/PRE_ENT_CAMPAIGN_PROGRESS.md).
- Wave 5 checkpoint on [`T-IGX-034-A.md`](T-IGX-034-A.md).
- This ticket file (scaffold + plan digest).

**Base SHA:** `6399a16574e3784261dd2473770f3bb8dc922716` (`development` / `origin/development` at Wave 5 entry). **AUDIT_HEAD:** `1e83ade3f5fb951a9eedd85d4dae89ead48955ee`. **CERT_DOCS_HEAD:** `d00df592037404cd6c2b1bc95c00757544d551d8`. **WAVE4_PUBLICATION_SHA:** `c4a3b751bf7d6716bb0003bed23ae20b1e8f910f`.

## Plan digest

Plan digest: c7822cbdd5da68e5a882d64b275a17357535f7b94fd34abc61927866d86fe227

## Approved

Operator authorization (chat 2026-11-10 — recorded verbatim; implementer unsigned):

```text
Approved: D-W6-01=a; D-W6-02=a; D-W6-03=a; D-W6-04=b; D-W6-05=a; D-W6-06=b; D-W6-07=a
Zatwierdzam terminalne rozliczenie kampanii PRE-ENT z wynikiem PRE_ENT_PREDECESSORS_EXACT_BLOCKERS.
```

Wave 5 publication (prior chat — PR #439 merged):

```text
Zatwierdzam formalną publikację Wave 5 zgodnie z T-IGX-034-I, pod warunkiem zachowania wszystkich rzeczywistych ograniczeń i statusów EXACT_BLOCKER.
D-W5-PUB=merge
```

## Test contract

```text
uv run python scripts/governance/verify_iiac_policy_activation.py --require-in-force
```

Before push (if operator approval line present): `uv run python -m scripts.governance.igx_gov ticket` (profile `fast`).

## Out of scope

- Product paths (`intergrax/`, `agents/`, `applications/`, `scripts/`).
- FRZ promotion, G6 campaign PASS, `ARCH-FREEZE` stage closure, Population E qualification, ACCEPT_SCOPED → full cert promotion.
- Wave 6 cross-audit execution, campaign terminal label (`ALL_PRE_ENT_PREDECESSORS_INDEPENDENTLY_CERTIFIED` / `PRE_ENT_PREDECESSORS_EXACT_BLOCKERS`).
- Merge to `development`, push, or PR merge unless operator provides explicit approval line (`Approved: …` or `D-W5-PUB=merge`).
- Tickets T-IGX-034-C, T-IGX-034-D, T-IGX-034-H (no new scaffolds).
- Commit of `tests/qualification/ent_audit_x/CANDIDATE_ENTERPRISE_AUDIT_BASELINE.json`.

## Lifecycle state

closed

## Checkpoint

Scaffold + matrix publication batch on branch `pre-ent/wave5-matrix-034i` (PR [#439](https://github.com/jakbuczarnecki/intergrax/pull/439)).

**Mechanical plan gateway:** `verify_plan_approval.py` requires a line starting `Approved:` — **not satisfied** without operator ISO line; chat authorization recorded above; lifecycle remains `plan_pending` per `governance/tickets/README.md`.

```text
uv run python scripts/governance/verify_plan_approval.py governance/tickets/T-IGX-034-I.md --plan <wave_5_matrix_closure_ab84abea.plan.md>
[ERROR] ticket missing Approved: line
exit 1 (2026-11-10 — expected; no forgery)
```

**Post-merge verification (on `origin/development` HEAD):** `verify_iiac_policy_activation.py --require-in-force`; `igx_gov ticket` — record exit codes in parent session report.

### W6-G4 closure (2026-11-10)

Parent `T-IGX-034-A` **closed** — campaign **`CLOSED — EXACT_BLOCKERS`**. Mechanical `verify_plan_approval` satisfied via operator `Approved:` line above (D-W6-05=a). **No** FRZ / §3 / full cert promotion.
